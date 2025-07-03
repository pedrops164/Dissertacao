from typing import Tuple
from prompts import get_hyde_rag_prompt
from config import config
from vectordb import VectorDB
from llm_system import LLMRAGSystem
from llm_client import NebiusLLMClient

class HyDERAGSystem(LLMRAGSystem):
    def __init__(self, system_name: str, llm_client: NebiusLLMClient, vector_db: VectorDB, rag_k: int):
        super().__init__(system_name, llm_client, vector_db, rag_k)

    def query(self, prompt: str, formatted_prompt: str) -> Tuple[str, dict]:
        """
        Simulate response generation with Hyde-RAG.
        This function is a placeholder and should be replaced with actual implementation.
        """

        hyde_prompt = get_hyde_rag_prompt(prompt)
        hypothetical_doc, _ = self.llm_client.call_llm_assessment(prompt=hyde_prompt)  # Simulate LLM generating a hypothetical document

        # retrieve relevant documents from the vector database
        retrieved_docs = self.vector_db.retrieve_context(hypothetical_doc, n_results=self.rag_k)
        # build context from retrieved documents
        context = "\n\n---\n\n".join(retrieved_docs) if retrieved_docs else ""
        
        response, tokens_count = self.llm_client.query_llm_with_context(formatted_prompt, context)
        return response, {
            "tokens_count": tokens_count,
            "retrieved_docs": retrieved_docs,
            "hypothetical_doc": hypothetical_doc
        }