import time
from llm_client import call_llm_assessment, query_llm_with_context
from vector_database import retrieve_context # Hybrid search + RRF context retrieval
from prompts import get_self_rag_retrieval_prompt, get_self_rag_critique_prompt, get_self_rag_generation_prompt_message, get_self_rag_critique_answer_prompt
from config import config

# --- Configuration ---
# How many initial documents to retrieve before critique/filtering
SELF_RAG_INITIAL_K = config.get("SELF_RAG_INITIAL_K")
# How many relevant documents to finally use for generation context
SELF_RAG_FINAL_CONTEXT_K = config.get("RAG_FINAL_CONTEXT_K")

# --- Self-RAG Core Logic ---

def generate_response_self_rag(query):
    """
    Generates a response using a simulated Self-RAG workflow.

    Steps:
    1. Decide if retrieval is necessary.
    2. Retrieve documents if needed.
    3. Critique retrieved documents for relevance.
    4. Generate an answer based on relevant documents (or none).
    5. Critique the generated answer for factuality/relevance.
    6. Return the answer and critique status.
    """
    print(f"\n--- Starting Self-RAG Process for Query: '{query}' ---")
    start_time = time.time()
    tokens_count = 0

    # --- Step 1: Decide if Retrieval is Necessary ---
    print("\n[Step 1/5] Deciding if retrieval is needed...")
    retrieval_prompt = get_self_rag_retrieval_prompt(query)
    decision, n_tokens = call_llm_assessment(retrieval_prompt, max_tokens=10)
    tokens_count += n_tokens
    print(f"  > LLM Decision on Retrieval: {decision}")

    needs_retrieval = False
    if decision and "YES" in decision.upper():
        needs_retrieval = True
        print("  > Conclusion: Retrieval IS needed.")
    else:
        print("  > Conclusion: Retrieval is likely NOT needed.")

    retrieved_docs = []
    relevant_docs = []
    filtered_context = ""

    # --- Step 2: Retrieve Documents (if needed) ---
    if needs_retrieval:
        print(f"\n[Step 2/5] Retrieving Top-{SELF_RAG_INITIAL_K} documents (Hybrid Search + RRF)...")
        retrieved_docs = retrieve_context(query, n_results=SELF_RAG_INITIAL_K) # Fetch more

        if not retrieved_docs:
             print("  > Warning: Retrieval needed but no documents found.")
             needs_retrieval = False # Cannot proceed with critique/context

    # --- Step 3: Critique Retrieved Documents (if needed & available) ---
    if needs_retrieval and retrieved_docs:
        print(f"\n[Step 3/5] Critiquing {len(retrieved_docs)} retrieved documents...")
        relevant_docs = []
        for i, doc_text in enumerate(retrieved_docs):
            if not doc_text: continue
            critique_prompt = get_self_rag_critique_prompt(query, doc_text)
            critique, n_tokens = call_llm_assessment(critique_prompt, max_tokens=10)
            tokens_count += n_tokens
            print(f"  > Critiquing Doc {i+1}: Result = {critique}")
            if critique and "IRRELEVANT" not in critique.upper():
                relevant_docs.append(doc_text)
                print(f"    >> Doc {i+1} kept.")
            # Early exit if we have enough relevant docs
            if len(relevant_docs) >= SELF_RAG_FINAL_CONTEXT_K:
                 print(f"  > Found sufficient ({len(relevant_docs)}) relevant documents. Stopping critique early.")
                 break

        if not relevant_docs:
             print("  > No relevant documents found after critique.")
        else:
             print(f"  > Selected {len(relevant_docs)} relevant documents for context.")
             # Prepare final context string from relevant docs
             filtered_context = "\n\n---\n\n".join(relevant_docs[:SELF_RAG_FINAL_CONTEXT_K])

    # --- Step 4: Generate Answer ---
    print("\n[Step 4/5] Generating answer...")
    generated_answer, n_tokens = query_llm_with_context(query, filtered_context)
    tokens_count += n_tokens

    # --- Step 5: Critique Generated Answer (Self-Reflection) ---
    print("\n[Step 5/5] Critiquing generated answer...")
    critique_answer_prompt = get_self_rag_critique_answer_prompt(query, filtered_context, generated_answer)
    final_critique, n_tokens = call_llm_assessment(critique_answer_prompt, max_tokens=10)
    tokens_count += n_tokens
    print(f"  > Final Answer Critique Result: {final_critique}")

    end_time = time.time()
    print(f"\n--- Self-RAG Process Completed in {end_time - start_time:.2f} seconds ---")

    # --- Step 6: Return Results ---
    # Normalize critique result
    if final_critique and "SUPPORTED" in final_critique.upper():
        critique_status = "SUPPORTED"
    elif final_critique and "CONTRADICTORY" in final_critique.upper():
         critique_status = "CONTRADICTORY"
    elif final_critique and "NOT_SUPPORTED" in final_critique.upper():
         critique_status = "NOT_SUPPORTED"
    elif final_critique and "NOT_APPLICABLE" in final_critique.upper():
         critique_status = "NOT_APPLICABLE" # No context was used
    else:
         critique_status = "UNKNOWN" # LLM critique failed or format incorrect

    return {
        "query": query,
        "needs_retrieval": needs_retrieval,
        "num_retrieved": len(retrieved_docs),
        "num_relevant": len(relevant_docs),
        "final_context_used": bool(filtered_context),
        "generated_answer": generated_answer,
        "answer_critique": critique_status,
        "tokens_count": tokens_count,
    }

# --- Example Usage ---
if __name__ == "__main__":
    # Ensure vector_database.py setup ran successfully first if retrieval is needed
    # (especially the hybrid search part with BM25 index)

    test_queries = [
        "De que forma a obra de Fernando Pessoa, particularmente através dos seus heterónimos, explora a fragmentação da identidade e a modernidade em Portugal no início do século XX?",
        "Compare as abordagens e os impactos das políticas económicas implementadas em Portugal durante o Estado Novo com as políticas adotadas após a adesão à CEE (atual União Europeia).",
    ]

    for q in test_queries:
        result = generate_response_self_rag(q)
        print("\n" + "="*50)
        print(f"Query: {result['query']}")
        print(f"Generated Answer: {result['generated_answer']}")
        print(f"Retrieval Needed? {result['needs_retrieval']}")
        if result['needs_retrieval']:
             print(f"Docs Retrieved/Relevant: {result['num_retrieved']} / {result['num_relevant']}")
        print(f"Final Context Used? {result['final_context_used']}")
        print(f"Answer Critique Status: {result['answer_critique']}")
        print("="*50 + "\n")