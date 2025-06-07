
from vector_database import vector_db # <--- Import the retrieval function
from config import config
from llm_client import query_llm_with_context
# Fusion Configuration
FUSION_RRF_K = 60 # RRF k parameter (smoothing factor, typically 60)~
# How many documents to retrieve from each retriever before fusion
FUSION_RAG_INITIAL_K = config.FUSION_RAG_INITIAL_K
# How many relevant documents to finally use for generation context
RAG_FINAL_CONTEXT_K = config.RAG_FINAL_CONTEXT_K

# --- Reciprocal Rank Fusion (RRF) ---
def reciprocal_rank_fusion(list_of_ranked_lists, k=FUSION_RRF_K):
    """
    Performs RRF on multiple ranked lists of documents (strings).
    Args:
        list_of_ranked_lists: A list where each element is a list of document strings,
                              ordered by relevance from a single retriever.
        k: The RRF smoothing parameter (default 60).
    Returns:
        A single list of document strings, re-ranked by RRF score.
    """
    if not list_of_ranked_lists:
        return []

    rrf_scores = {} # Dictionary to store RRF scores: {doc_text: score}

    print(f"\nPerforming RRF fusion on {len(list_of_ranked_lists)} result lists...")
    for ranked_list in list_of_ranked_lists:
        if not isinstance(ranked_list, list):
             print(f"Warning: Skipping invalid item in fusion input (expected list, got {type(ranked_list)}).")
             continue
        for rank, doc_text in enumerate(ranked_list):
            if not isinstance(doc_text, str): # Ensure we are dealing with strings
                 print(f"Warning: Skipping invalid document item (expected str, got {type(doc_text)}) at rank {rank}.")
                 continue
                 
            # Calculate RRF score increment: 1 / (k + rank + 1) (rank is 0-based)
            score_increment = 1.0 / (k + rank + 1)
            if doc_text in rrf_scores:
                rrf_scores[doc_text] += score_increment
            else:
                rrf_scores[doc_text] = score_increment

    # Sort documents based on their aggregated RRF score (descending)
    sorted_docs = sorted(rrf_scores.keys(), key=lambda doc: rrf_scores[doc], reverse=True)
    print(f"RRF fusion completed. Produced {len(sorted_docs)} unique ranked documents.")
    return sorted_docs


# --- Main retrieval function using Hybrid Search + RRF ---
def retrieve_context_fusion(query, final_k=RAG_FINAL_CONTEXT_K, retrieval_k=FUSION_RAG_INITIAL_K):
    """
    Retrieves relevant context using hybrid search (Vector + BM25) and RRF fusion.
    """
    if vector_db is None:
        print("Error: Hybrid VectorDatabase is not initialized. Cannot retrieve context.")
        return ""

    print(f"\n--- Starting Hybrid Retrieval for query: '{query}' ---")
    print(f"(Fetching Top {retrieval_k} from Vector and BM25, fusing to Top {RAG_FINAL_CONTEXT_K})")

    # 1. Perform individual searches
    vector_results = vector_db.search_vector(query, n_results=retrieval_k)
    bm25_results = vector_db.bm25_index.search_bm25(query, n_results=retrieval_k)

    # 2. Fuse the results using RRF
    fused_results = reciprocal_rank_fusion([vector_results, bm25_results])

    # 3. Select top N results after fusion
    final_context_docs = fused_results[:final_k]

    # 4. Combine into a single context string
    context = "\n\n---\n\n".join(final_context_docs)
    if not context:
        print("Hybrid retrieval and fusion returned no relevant context.")
    else:
        print(f"\n--- Generated Fused Context (Top {len(final_context_docs)}) ---")
        #print(context) # Optionally print the full context
        print("--- End Hybrid Retrieval ---")
        
    return context

def generate_response_fusion_rag(query):
    """
    Generate a response based on the query using RAG.
    Retrieves context from the vector database before calling the LLM.

    Args:
        query (str): User query

    Returns:
        str: Generated response
    """
    # Retrieve relevant context from the vector database
    print(f"Retrieving context for query: {query}")
    context = retrieve_context_fusion(query) # Retrieve top 3 chunks
    if not context:
        print("Warning: No relevant context found.")
        # Optional: Handle case where no context is found (e.g., fall back to no-RAG or inform the LLM)
        context = "No specific context was found for this query." # Provide default text

    print(f"Retrieved Context:\n---\n{context}...\n---")

    return query_llm_with_context(query, context)

if __name__ == "__main__":
    # Example usage (ensure vector_database.py ran its setup)
    query1 = "Has amantadine ER been approved by the FDA?"
    response1 = generate_response_fusion_rag(query1)
    print(f"\nQuery: {query1}")
    print(f"Hybrid RAG Response:\n{response1}")

    query2 = "Can PRL3-zumab inhibit PRL3+ cancer cells in vitro and in vivo?"
    response2 = generate_response_fusion_rag(query2)
    print(f"\nQuery: {query2}")
    print(f"Hybrid RAG Response:\n{response2}")

    query3 = "Does Estrogen lead to forkhead FoxA1 activation?"
    response3 = generate_response_fusion_rag(query3)
    print(f"\nQuery: {query3}")
    print(f"Hybrid RAG Response:\n{response3}")