# Implementation of Corrective RAG using web search as fallback mechanism if the retrieved context isn't sufficient.

from vector_database import vector_db, DEFAULT_N_RESULTS, retrieve_context # <--- Import the retrieval function
from simple_rag import generate_response_rag # <--- Import the simple RAG function
from llm_client import call_llm_assessment # Your LLM client setup
from websearch_tavily import search_tavily # Import Tavily client
import re
import time
import logging
import json
from prompts import get_crag_assessment_prompt, get_crag_rewrite_prompt
from typing import Tuple

# --- Logging Setup ---
# Create logger
logger = logging.getLogger('CorrectiveRAG')
logger.setLevel(logging.INFO) # Set minimum level to log

# Prevent adding multiple handlers if this script/module is reloaded
if not logger.handlers:
    # Create file handler
    log_file = 'rag_execution.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Create console handler (optional, for warnings/errors)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING) # Only show warnings and errors on console

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler) # Add console handler

logger.info("--- Logger Initialized ---")

#assessment_prompt = get_crag_prompts_en()

def assess_context_relevance(query, context=None):
    """
    Assess if the retrieved context is relevant and sufficient, deciding
    whether web search is needed.

    Args:
        query (str): User query.
        context (str): Context retrieved from the vector database.

    Returns:
        int: An integer indicating the decision:
             1 -> Context is insufficient/irrelevant. Use web search ONLY.
             2 -> Context is relevant but potentially incomplete/stale. Use BOTH web search AND retrieved context.
             3 -> Context is relevant and sufficient. Use retrieved context ONLY.
             Returns 2 as a fallback if the LLM assessment fails.
             Returns 1 if the initial context is empty/None.
    """

    # Handle cases where no context was retrieved initially
    if not context: # Check if there's no context
        logger.info("No meaningful context retrieved from vector DB. Decision: 1 (Web Search ONLY).")
        return 1, 0

    # Limit context length for the assessment prompt
    context_snippet = context[:2000] # Use first 2000 chars for assessment

    assessment_prompt = get_crag_assessment_prompt(query, context_snippet)

    decision_str, total_tokens = call_llm_assessment(assessment_prompt)

    # Parse the LLM decision
    if decision_str:
        # Try to find the number directly
        match = re.search(r'\b([123])\b', decision_str)
        if match:
            decision_int = int(match.group(1))
            logger.info(f"Parsed assessment decision: {decision_int}")
            # Log detailed conclusion based on decision
            if decision_int == 1: logger.info("Assessment Conclusion: Context irrelevant/insufficient or query needs real-time data. Use Web Search ONLY.")
            elif decision_int == 2: logger.info("Assessment Conclusion: Context relevant but needs supplement. Use BOTH.")
            else: logger.info("Assessment Conclusion: Context sufficient. Use Local Context ONLY.")
            return decision_int, total_tokens
        else:
            logger.warning(f"Could not parse LLM assessment decision ('{decision_str}'). Defaulting to 2 (Use BOTH).")
            return 2, total_tokens # Fallback to using both if parsing fails
    else:
        logger.warning("LLM assessment call failed. Defaulting to 2 (Use BOTH).")
        return 2, total_tokens # Fallback to using both if LLM call fails

def rewrite_query_for_websearch(query):
    """
    Rewrites the user query using an LLM to be more effective for web search.
    Falls back to the original query if the LLM call fails.

    Args:
        query (str): Original user query.

    Returns:
        str: Rewritten query optimized for web search, or the original query on failure.
    """

    rewrite_prompt = get_crag_rewrite_prompt(query)

    # Use the generic LLM helper
    rewritten_query, total_tokens = call_llm_assessment(rewrite_prompt, max_tokens=60, temperature=0.0)

    if rewritten_query:
        # Remove potential quotes LLM might add
        rewritten_query = rewritten_query.strip().strip('"')
        logger.info(f"Query rewritten for web search: '{rewritten_query}'")
        return rewritten_query, total_tokens
    else:
        logger.warning("LLM rewrite failed. Using original query for web search.")
        return query, 0 # Fallback to original query


def perform_web_search(query):
    """
    Performs a web search using the Tavily API based on the query.

    Args:
        query (str): The query string optimized for web search.

    Returns:
        str: A string containing concatenated relevant snippets from search results,
             or an empty string if the search fails or yields no useful results.
    """
    context = search_tavily(query)
    return context # Return the concatenated context from the search results

def generate_response_corrective_rag(query) -> Tuple[str, int]:
    """
    Generates a response using Corrective RAG. This process involves:
    1. Retrieving initial context from a vector database.
    2. Assessing the relevance and sufficiency of the retrieved context using an LLM.
    3. Based on the assessment, deciding whether to use only local context,
       only web search, or both.
    4. Performing a web search if needed (potentially after rewriting the query).
    5. Combining the necessary contexts.
    6. Generating the final response using an LLM with the compiled context.
    7. Logs detailed execution information in a structured format.

    Args:
        query (str): The user query.

    Returns:
        str: The generated response.
    """
    # Dictionary to store detailed execution information for logging
    execution_data = {
        "query": query,
        "start_time": time.time(),
        "start_time_iso": time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        "local_context_retrieved": None,
        "local_context_length": 0,
        "assessment_decision": None,
        "websearch_performed": False,
        "rewritten_websearch_query": None,
        "websearch_context_retrieved": None,
        "websearch_context_length": 0,
        "final_context_source": "None",
        "final_context_used": None,
        "final_context_length": 0,
        "generated_response": None,
        "end_time": None,
        "end_time_iso": None,
        "duration_seconds": None,
        "status": "In Progress"
    }
    tokens_count = 0

    # Step 1: Retrieve initial context from the vector database
    print("\n[Step 1/5] Retrieving initial context from Vector DB...")
    logger.info("Attempting to retrieve initial context from Vector DB.")
    local_context = retrieve_context(query) # Retrieve top N chunks based on similarity
    local_context = "\n\n---\n\n".join(local_context) # Join the retrieved chunks into a single string
    execution_data["local_context_retrieved"] = local_context
    execution_data["local_context_length"] = len(local_context) if local_context else 0

    if local_context:
        logger.info(f"Successfully retrieved local context (length: {len(local_context)} chars).")
    else:
        logger.info("No initial context found in Vector DB.")

    # Step 2: Assess if web search is needed based on the query and initial context
    print("\n[Step 2/5] Assessing context relevance and web search need...")
    context_relevance_decision, n_tokens = assess_context_relevance(query, local_context)
    tokens_count += n_tokens
    execution_data["assessment_decision"] = context_relevance_decision
    logger.info(f"Context relevance assessment decision: {context_relevance_decision}")

    # Step 3: Perform web search if the assessment indicated it's necessary (Decisions 1 or 2)
    print("\n[Step 3/5] Performing web search if required...")
    websearch_context = "" # Initialize websearch_context to empty string
    execution_data["websearch_performed"] = (context_relevance_decision == 1 or context_relevance_decision == 2)

    if execution_data["websearch_performed"]:
        logger.info("Assessment decision requires web search (Decision 1 or 2).")
        # Rewrite the query for web search
        websearch_query, n_tokens = rewrite_query_for_websearch(query)
        tokens_count += n_tokens
        execution_data["rewritten_websearch_query"] = websearch_query

        # Perform the actual web search
        websearch_context = perform_web_search(websearch_query)
        execution_data["websearch_context_retrieved"] = websearch_context
        execution_data["websearch_context_length"] = len(websearch_context) if websearch_context else 0
    else:
        logger.info("Assessment decision does not require web search (Decision 3). Skipping web search.")


    # Step 4: Prepare the final context based on the assessment decision and search results
    print("\n[Step 4/5] Compiling final context for response generation...")
    final_context = ""
    context_source_description = "No context available" # Default description

    if context_relevance_decision == 1:
        # Use Web Search Only
        if websearch_context:
            final_context = f"CONTEXT FROM WEB SEARCH:\n{websearch_context}"
            context_source_description = "Web Search Only"
        else:
            # If web search was supposed to be used but yielded nothing
            logger.warning("Web search selected (Decision 1) but returned no context.")
            final_context = "Web search was attempted but returned no information."
            context_source_description = "Web Search attempted, no results"

    elif context_relevance_decision == 2:
        # Use Both Local Context and Web Search
        logger.info("Compiling final context: Use Both Local and Web Search.")
        if local_context and websearch_context:
            final_context = f"CONTEXT FROM LOCAL DATABASE:\n{local_context}\n\n---\n\nCONTEXT FROM WEB SEARCH:\n{websearch_context}"
            context_source_description = "Local and Web Search"
        elif local_context:
            # Web search was intended but failed or returned nothing
            final_context = f"CONTEXT FROM LOCAL DATABASE:\n{local_context}\n\n---\n\nCONTEXT FROM WEB SEARCH:\n(Web search yielded no results or failed)"
            logger.warning("Web search failed or returned nothing when using both was decided.")
            context_source_description = "Local Only (Web Search failed)"
        elif websearch_context:
            # Local context was empty, but web search succeeded
            final_context = f"CONTEXT FROM LOCAL DATABASE:\n(No relevant local context found)\n\n---\n\nCONTEXT FROM WEB SEARCH:\n{websearch_context}"
            logger.warning("No local context found when using both was decided.")
            context_source_description = "Web Search Only (Local context empty)"
        else:
             # Neither local nor web search provided context
            logger.warning("Neither local nor web search provided context when using both was decided.")
            final_context = "Neither local context nor web search provided information for this query."
            context_source_description = "Neither source provided context"

    elif context_relevance_decision == 3:
        # Use Local Context Only
        logger.info("Compiling final context: Use Local Context Only.")
        if local_context:
            final_context = local_context
            context_source_description = "Local Context Only"
        else:
            # Local context was selected but was empty (should have been caught by initial check, but double check)
            logger.warning("Local context selected (Decision 3) but was empty.")
            final_context = "No specific context was found for this query."
            context_source_description = "Local Only (context empty)"

    # Log the final context state and length
    execution_data["final_context_source"] = context_source_description

    if final_context and final_context not in [
        "Web search was attempted but returned no information.",
        "Neither local context nor web search provided information for this query.",
        "No specific context was found for this query."
        ]:
        logger.info(f"Final context prepared (length: {len(final_context)} chars). Source: {context_source_description}")
        execution_data["final_context_used"] = final_context
        execution_data["final_context_length"] = len(final_context)
    else:
        logger.warning("No usable final context available for generation.")
        # Ensure final_context is not empty if we intend to pass it to the LLM
        # Passing a specific message ensures the LLM knows no info was found.
        if not final_context:
             final_context = "No relevant information could be found to answer the query."
             execution_data["final_context_used"] = final_context
             execution_data["final_context_length"] = len(final_context)


    # Step 5: Generate the final response using the compiled context
    print("\n[Step 5/5] Generating final response...")
    logger.info("Generating final response using the compiled context.")
    # Call the simple RAG function with the compiled final context and the original query
    response, n_tokens = generate_response_rag(query, final_context)
    tokens_count += n_tokens
    execution_data["generated_response"] = response


    # Log detailed execution data in a structured format (JSON)
    execution_data["end_time"] = time.time()
    execution_data["end_time_iso"] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    execution_data["duration_seconds"] = execution_data["end_time"] - execution_data["start_time"]
    execution_data["status"] = "Completed"

    # Log the structured data
    # Use json.dumps to serialize the dictionary to a JSON string
    # Use ensure_ascii=False to correctly handle non-ASCII characters in contexts/response
    try:
        logger.info(json.dumps(execution_data, indent=4, ensure_ascii=False, sort_keys=True))
    except Exception as e:
        logger.error(f"Failed to log structured execution data: {e}")
        # Log essential data even if full JSON fails
        logger.info(f"Fallback Log: Query: {query}, Status: {execution_data['status']}, Duration: {execution_data['duration_seconds']:.2f}s, Decision: {execution_data['assessment_decision']}, Local Len: {execution_data['local_context_length']}, Web Len: {execution_data['websearch_context_length']}, Final Len: {execution_data['final_context_length']}, Final Source: {execution_data['final_context_source']}")


    # Print completion message
    logger.info(f"--- Corrective RAG process completed for query: '{query}' ---")
    logger.info(f"Generated response (first 200 chars): {response[:200]}...") # Log snippet of response

    return response, tokens_count

# --- Example Usage ---
if __name__ == "__main__":
    # Example Queries:
    test_queries = [
        # Query likely answerable by vector DB (if populated with relevant data)
        "Explain the concept of 'Saudade' in the Portuguese culture.",
        # Query likely needing recent info / web search
        "What were the results of the last Portuguese elections?",
         # Query requiring current information not likely in a static DB
        "What is the weather like in Lisbon tomorrow?",
        # Query that might benefit from both
        "Compare the Portuguese economy in 2010 and currently."
    ]

    for i, q in enumerate(test_queries):
        print(f"\n" + "="*60)
        print(f"Processing Query {i+1}/{len(test_queries)}:")
        print(f"Query: {q}")
        print("="*60 + "\n")
        logger.info(f"Processing Query {i+1}/{len(test_queries)}: '{q}'")

        # Make sure vector_database, simple_rag, llm_client, and websearch_tavily
        # modules are accessible and configured correctly.
        # The vector DB should be populated if local context scenarios are to be tested effectively.
        result = generate_response_corrective_rag(q)

        # Print the final result clearly to console after the process finishes
        print("\n" + "-"*60)
        print(f"GENERATED ANSWER FOR QUERY {i+1}:")
        print(f"{result}")
        print("="*60 + "\n")


    print("Corrective RAG Example Usage Finished.")
    logger.info("Corrective RAG Example Usage Finished.")