from config import config
from vectordb import vector_db
from typing import Tuple
from llm_system import LLMSystem
from llm_client import NebiusLLMClient

RAG_FINAL_CONTEXT_K = config.get("RAG_FINAL_CONTEXT_K")
    
class SimpleRAGSystem(LLMSystem):
    def __init__(self, system_name: str, llm_client: NebiusLLMClient):
        super().__init__(system_name, llm_client)

    def query(self, prompt: str, formatted_prompt: str) -> Tuple[str, dict]:
        """
        Generate a response based on the query using RAG.
        Retrieves context from the vector database before calling the LLM.

        Args:
            query (str): User query

        Returns:
            str: Generated response
        """
        # retrieve relevant documents from the vector database
        retrieved_docs = vector_db.retrieve_context(prompt, n_results=RAG_FINAL_CONTEXT_K)
        # build context from retrieved documents
        context = "\n\n---\n\n".join(retrieved_docs) if retrieved_docs else ""
        
        response, tokens_used = self.llm_client.query_llm_with_context(formatted_prompt, context)
        metrics = {
            "token_count": tokens_used,
            "retrieved_docs_count": len(retrieved_docs),
            "retrieved_docs": retrieved_docs,
        }
        return response, metrics