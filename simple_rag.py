from config import config
from vectordb import VectorDB
from typing import Tuple
from llm_system import LLMRAGSystem
from llm_client import NebiusLLMClient
    
class SimpleRAGSystem(LLMRAGSystem):
    def __init__(self, system_name: str, llm_client: NebiusLLMClient, vector_db: VectorDB, rag_k: int):
        super().__init__(system_name, llm_client, vector_db, rag_k)

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
        retrieved_docs = self.vector_db.retrieve_context(prompt, n_results=self.rag_k)
        # build context from retrieved documents
        context = "\n\n---\n\n".join(retrieved_docs) if retrieved_docs else ""
        
        response, tokens_used = self.llm_client.query_llm_with_context(formatted_prompt, context)
        metrics = {
            "token_count": tokens_used,
            "retrieved_docs_count": len(retrieved_docs),
            "retrieved_docs": retrieved_docs,
        }
        return response, metrics