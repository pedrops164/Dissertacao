from llm_client import query_llm_with_context
from config import config
from vectordb import vector_db

RAG_FINAL_CONTEXT_K = config.get("RAG_FINAL_CONTEXT_K")

def generate_response_simple_rag(query):
    """
    Generate a response based on the query using RAG.
    Retrieves context from the vector database before calling the LLM.

    Args:
        query (str): User query

    Returns:
        str: Generated response
    """
    # retrieve relevant documents from the vector database
    retrieved_docs = vector_db.retrieve_context(query, n_results=RAG_FINAL_CONTEXT_K)
    # build context from retrieved documents
    context = "\n\n---\n\n".join(retrieved_docs) if retrieved_docs else ""
    
    return query_llm_with_context(query, context)