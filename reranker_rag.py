import time
from vectordb import VectorDB
from config import config
from llm_system import LLMRAGSystem
from llm_client import NebiusLLMClient
from typing import Tuple

class RerankerRAGSystem(LLMRAGSystem):
    def __init__(self, system_name: str, llm_client: NebiusLLMClient, vector_db: VectorDB, rag_k: int):
        super().__init__(system_name, llm_client, vector_db, rag_k)

    def query(self, prompt: str, formatted_prompt: str) -> Tuple[str, dict]:
        """
        Generates a response using a RAG pipeline with a reranking step.

        Steps:
        1. Retrieve initial set of documents.
        2. Rerank the retrieved documents.
        3. Select top N reranked documents for context.
        4. Generate an answer based on the selected context.
        5. Return the answer.
        """
        #print(f"\n--- Starting RAG with Reranker Process for Query: '{prompt}' ---")
        start_time = time.time()
        tokens_count = 0 # Assuming your llm_client returns token counts

        initial_k = self.rag_k * 2
        # --- Step 1: Retrieve Initial Documents ---
        #print(f"\n[Step 1/4] Retrieving Top-{initial_k} documents for reranking...")
        # This function should return a list of document texts
        # It might internally use your hybrid search + RRF
        initial_docs = self.vector_db.retrieve_context(prompt, n_results=initial_k)
        if not initial_docs:
            #print("  > No documents retrieved. Proceeding without context.")
            initial_docs = []
        #print(f"  > Retrieved {len(initial_docs)} candidate documents.")

        reranked_docs = []
        final_context = ""

        # --- Step 2: Rerank Retrieved Documents ---
        if initial_docs:
            #print(f"\n[Step 2/4] Reranking {len(initial_docs)} documents...")
            # The rerank_documents function should ideally interact with a reranking model
            reranked_docs_full_list = self.rerank_documents(prompt, initial_docs)
            # Select the top N documents after reranking
            reranked_docs = reranked_docs_full_list[:self.rag_k]
            #print(f"  > Selected Top-{len(reranked_docs)} documents after reranking.")

            if not reranked_docs:
                raise ValueError("No documents remained after reranking. Cannot proceed with answer generation.")
            else:
                # Prepare final context string from reranked top N docs
                final_context = "\n\n---\n\n".join(reranked_docs)
        else:
            #print("\n[Step 2/4] Skipping reranking as no documents were initially retrieved.")
            pass


        # --- Step 3: Generate Answer ---
        #print("\n[Step 3/4] Generating answer...")
        # Use your existing prompt engineering, passing the refined context
        generated_answer, n_tokens = self.llm_client.query_llm_with_context(formatted_prompt, final_context)
        tokens_count += n_tokens

        # --- Step 4: Return Results ---
        end_time = time.time()
        #print(f"\n--- RAG with Reranker Process Completed in {end_time - start_time:.2f} seconds ---")

        return generated_answer, {
            "docs_initially_retrieved_count": len(initial_docs),
            "reranked_and_selected_count": len(reranked_docs),
            "reranked_docs": reranked_docs,
            "tokens_count": tokens_count,
        }
    
    def rerank_documents(self, query: str, documents: list[str]) -> list[str]:
        """
        A simple reranking method based on keyword matching, position, and frequency.

        Args:
            query (str): User query.
            documents (list[str]): A list of document text strings from initial retrieval.

        Returns:
            list[str]: Reranked list of document text strings. The full list is returned,
                    reordered. The calling function will select the top N.
        """
        # Extract important keywords from the query (simple split, lowercasing, filter short words)
        query_keywords = [word.lower() for word in query.split() if len(word) > 3]

        if not query_keywords:
            #print("    >> No useful keywords (len > 3) extracted from query. Returning documents in original order.")
            return documents
        
        if not documents:
            #print("    >> No documents provided to rerank. Returning empty list.")
            return []

        scored_documents = []

        for i, doc_text in enumerate(documents):
            # Ensure doc_text is a string and not empty, otherwise assign a very low score.
            if not doc_text or not isinstance(doc_text, str):
                #print(f"    >> Warning: Document at index {i} is empty or not a string. Assigning minimal score.")
                scored_documents.append({"text": doc_text, "relevance_score": -1.0, "original_index": i})
                continue

            doc_lower = doc_text.lower()
            current_doc_score = 0.0

            for keyword in query_keywords:
                if keyword in doc_lower:
                    current_doc_score += 1.0  # Base score for keyword presence

                    # Bonus if keyword appears near the beginning
                    first_position = doc_lower.find(keyword)
                    # Check if keyword is found and in the first quarter of the document
                    if first_position != -1 and first_position < (len(doc_lower) / 4):
                        current_doc_score += 0.5 # Bonus for early appearance

                    # Bonus for keyword frequency (capped)
                    frequency = doc_lower.count(keyword)
                    current_doc_score += min(0.2 * frequency, 1.0) # Cap frequency bonus (e.g., max +1.0 from frequency)

            scored_documents.append({"text": doc_text, "relevance_score": current_doc_score, "original_index": i})

        # Sort results by the calculated relevance score in descending order.
        # If scores are equal, maintain original relative order
        reranked_scored_documents = sorted(scored_documents, key=lambda x: x["relevance_score"], reverse=True)

        # Return just the text of the reranked documents
        final_reranked_texts = [doc["text"] for doc in reranked_scored_documents]

        return final_reranked_texts
