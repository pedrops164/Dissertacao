from abc import ABC, abstractmethod
from llm_client import NebiusLLMClient
from typing import Tuple
from vectordb import VectorDB

class LLMSystem(ABC):
    def __init__(self, system_name: str, llm_client: NebiusLLMClient):
        self.system_name = system_name
        self.llm_client = llm_client

    @abstractmethod
    def query(self, prompt: str, formatted_prompt: str) -> Tuple[str, dict]:
        pass

class LLMRAGSystem(LLMSystem):
    def __init__(self, system_name: str, llm_client: NebiusLLMClient, vector_db: VectorDB, rag_k: int):
        super().__init__(system_name, llm_client)
        self.vector_db = vector_db  # Vector database for RAG
        self.rag_k = rag_k  # Number of documents to retrieve for RAG

    @abstractmethod
    def query(self, prompt: str, formatted_prompt: str) -> Tuple[str, dict]:
        pass