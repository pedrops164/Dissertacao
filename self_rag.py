import re
import time
from llm_client import call_llm_assessment # Your LLM client setup
from vector_database import retrieve_context # Hybrid search + RRF context retrieval

# --- Configuration ---
# How many initial documents to retrieve before critique/filtering
SELF_RAG_RETRIEVAL_K = 7
# How many relevant documents to finally use for generation context
SELF_RAG_FINAL_CONTEXT_N = 3

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

    # --- Step 1: Decide if Retrieval is Necessary ---
    print("\n[Step 1/5] Deciding if retrieval is needed...")
    retrieval_prompt = f"""Does the following query likely require searching external documents for a factual and comprehensive answer, or can it be answered reliably from general knowledge?
Query: "{query}"
Answer ONLY with YES or NO."""
    decision = call_llm_assessment(retrieval_prompt, max_tokens=10)
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
        print(f"\n[Step 2/5] Retrieving Top-{SELF_RAG_RETRIEVAL_K} documents (Hybrid Search + RRF)...")
        # Use the existing hybrid retrieval, but get more docs initially
        # retrieve_context returns a single string, we need individual docs for critique.
        # We need to modify retrieve_context or add a function to get ranked docs *before* joining.
        # For now, let's SIMULATE getting individual docs by splitting the context.
        # THIS IS A HACK - ideally vector_database returns a list here.
        #temp_context = retrieve_context(query, n_results=SELF_RAG_RETRIEVAL_K, retrieval_k=SELF_RAG_RETRIEVAL_K * 2) # Fetch more
        temp_context = retrieve_context(query, n_results=SELF_RAG_RETRIEVAL_K) # Fetch more
        if temp_context:
             retrieved_docs = temp_context.split("\n\n---\n\n")
        print(f"  > Retrieved {len(retrieved_docs)} candidate documents.")

        if not retrieved_docs:
             print("  > Warning: Retrieval needed but no documents found.")
             needs_retrieval = False # Cannot proceed with critique/context

    # --- Step 3: Critique Retrieved Documents (if needed & available) ---
    if needs_retrieval and retrieved_docs:
        print(f"\n[Step 3/5] Critiquing {len(retrieved_docs)} retrieved documents...")
        relevant_docs = []
        for i, doc_text in enumerate(retrieved_docs):
            if not doc_text: continue
            critique_prompt = f"""Evaluate if the following document passage is relevant and helpful for answering the query. Consider if it directly addresses the query or provides useful background.
Query: "{query}"
Passage: "{doc_text[:1000]}" # Limit length for critique prompt
Answer ONLY with RELEVANT or IRRELEVANT."""
            critique = call_llm_assessment(critique_prompt, max_tokens=10)
            print(f"  > Critiquing Doc {i+1}: Result = {critique}")
            if critique and "IRRELEVANT" not in critique.upper():
                relevant_docs.append(doc_text)
                print(f"    >> Doc {i+1} kept.")
            # Early exit if we have enough relevant docs
            if len(relevant_docs) >= SELF_RAG_FINAL_CONTEXT_N:
                 print(f"  > Found sufficient ({len(relevant_docs)}) relevant documents. Stopping critique early.")
                 break

        if not relevant_docs:
             print("  > No relevant documents found after critique.")
        else:
             print(f"  > Selected {len(relevant_docs)} relevant documents for context.")
             # Prepare final context string from relevant docs
             filtered_context = "\n\n---\n\n".join(relevant_docs[:SELF_RAG_FINAL_CONTEXT_N])

    # --- Step 4: Generate Answer ---
    print("\n[Step 4/5] Generating answer...")
    generation_prompt_message = ""
    if filtered_context:
        print(f"  > Generating with {len(relevant_docs)} relevant document(s) as context.")
        system_prompt = "You are a helpful AI assistant. Answer the user's question based *only* on the provided relevant context. Be factual and concise."
        user_prompt = f"""Relevant Context:
{filtered_context}

Question: {query}

Based *only* on the relevant context provided, answer the question."""
        generation_prompt_message = [
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}
        ]
    else:
        print("  > Generating without external context (or no relevant context found).")
        system_prompt = "You are a helpful AI assistant. Answer the user's question directly using your general knowledge."
        user_prompt = f"Question: {query}"
        generation_prompt_message = [
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}
        ]

    try:
        generated_answer = call_llm_assessment(generation_prompt=generation_prompt_message, max_tokens=500, temperature=0.1)
        #response = openai_client.chat.completions.create(
        #    model="meta-llama/Llama-3.2-3B-Instruct", # Or your chosen model
        #    messages=generation_prompt_message,
        #    temperature=0.1, # Keep generation somewhat deterministic after critique
        #    max_tokens=500
        #)
        #generated_answer = response.choices[0].message.content.strip()
        print(f"  > Generated Answer (Initial): {generated_answer[:200]}...")
    except Exception as e:
         print(f"Error during LLM generation call: {e}")
         generated_answer = "[Error generating initial answer]"


    # --- Step 5: Critique Generated Answer (Self-Reflection) ---
    print("\n[Step 5/5] Critiquing generated answer...")
    critique_answer_prompt = f"""Evaluate the following generated answer based SOLELY on the provided query and context (if any).
Is the answer factually consistent with the context?
Is the answer relevant to the original query?

Query: "{query}"
Context Provided: "{filtered_context[:1000]}" # Limit context length
Generated Answer: "{generated_answer}"

Answer ONLY with one word: SUPPORTED, CONTRADICTORY, or NOT_SUPPORTED (if answer is irrelevant or not verifiable from context). If no context was provided, answer NOT_APPLICABLE."""

    final_critique = call_llm_assessment(critique_answer_prompt, max_tokens=10)
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
        "answer_critique": critique_status
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