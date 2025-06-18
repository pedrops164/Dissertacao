import time
from vectordb import vector_db
from prompts import get_self_rag_critique_prompt
from config import config
from llm_system import LLMSystem
from llm_client import NebiusLLMClient
from typing import Tuple

# --- Configuration ---
# How many initial documents to retrieve before critique/filtering
SELF_RAG_INITIAL_K = config.get("SELF_RAG_INITIAL_K")
# How many relevant documents to finally use for generation context
SELF_RAG_FINAL_CONTEXT_K = config.get("RAG_FINAL_CONTEXT_K")

class SelfRAGSystem(LLMSystem):
    def __init__(self, system_name: str, llm_client: NebiusLLMClient):
        super().__init__(system_name, llm_client)

    def query(self, prompt: str, formatted_prompt: str) -> Tuple[str, dict]:
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
        print(f"\n--- Starting Self-RAG Process for Query: '{prompt}' ---")
        start_time = time.time()
        tokens_count = 0

        retrieved_docs = []
        relevant_docs = []
        filtered_context = ""

        # --- Step 1: Retrieve Documents (if needed) ---
        print(f"\n[Step 1/3] Retrieving Top-{SELF_RAG_INITIAL_K} documents...")
        retrieved_docs = vector_db.retrieve_context(prompt, n_results=SELF_RAG_INITIAL_K) # Fetch more

        if retrieved_docs:
            # --- Step 2: Critique Retrieved Documents (if needed & available) ---
            print(f"\n[Step 2/3] Critiquing {len(retrieved_docs)} retrieved documents...")
            relevant_docs = []
            for i, doc_text in enumerate(retrieved_docs):
                if not doc_text: continue
                critique_prompt = get_self_rag_critique_prompt(prompt, doc_text)
                critique, n_tokens = self.llm_client.call_llm_assessment(critique_prompt)
                tokens_count += n_tokens
                print(f"  > Critiquing Doc {i+1}: Result = {critique}")
                if critique and "IRRELEVANT" not in critique.upper():
                    relevant_docs.append(doc_text)
                    #print(f"    >> Doc {i+1} kept.")
                # Early exit if we have enough relevant docs
                if len(relevant_docs) >= SELF_RAG_FINAL_CONTEXT_K:
                    #print(f"  > Found sufficient ({len(relevant_docs)}) relevant documents. Stopping critique early.")
                    break

            if relevant_docs:
                #print(f"  > Selected {len(relevant_docs)} relevant documents for context.")
                # Prepare final context string from relevant docs
                filtered_context = "\n\n---\n\n".join(relevant_docs[:SELF_RAG_FINAL_CONTEXT_K])

        # --- Step 4: Generate Answer ---
        print("\n[Step 3/3] Generating answer...")
        generated_answer, n_tokens = self.llm_client.query_llm_with_context(formatted_prompt, filtered_context)
        tokens_count += n_tokens

        end_time = time.time()
        print(f"\n--- Self-RAG Process Completed in {end_time - start_time:.2f} seconds ---")

        return generated_answer, {
            "retrieved_docs_count": len(retrieved_docs),
            "retrieved_docs": retrieved_docs,
            "relevant_docs_count": len(relevant_docs),
            "relevant_docs": relevant_docs,
            "tokens_count": tokens_count,
        }