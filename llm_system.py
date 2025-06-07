from abc import ABC, abstractmethod
from no_rag import generate_response_no_rag
from simple_rag import generate_response_simple_rag
from self_rag import generate_response_self_rag
from fusion_rag import generate_response_fusion_rag
from crag_rag import generate_response_corrective_rag
from reranker_rag import generate_response_reranker_rag
from hyde_rag import generate_response_hyde_rag
from typing import Tuple

class LLMSystem(ABC):
    def __init__(self, system_name: str):
        self.system_name = system_name

    @abstractmethod
    def query(self, prompt: str) -> Tuple[str, int]:
        pass

class NoRAGSystem(LLMSystem):
    def __init__(self, system_name: str):
        super().__init__(system_name)

    def query(self, prompt: str) -> Tuple[str, int]:
        # Simulate response generation without RAG
        return generate_response_no_rag(prompt)
    
class SimpleRAGSystem(LLMSystem):
    def __init__(self, system_name: str):
        super().__init__(system_name)

    def query(self, prompt: str) -> Tuple[str, int]:
        # Simulate response generation with simple RAG
        return generate_response_simple_rag(prompt)  # Assuming similar to NoRAG for now
    
class SelfRAGSystem(LLMSystem):
    def __init__(self, system_name: str):
        super().__init__(system_name)

    def query(self, prompt: str) -> Tuple[str, int]:
        # Simulate response generation with Self-RAG
        response =  generate_response_self_rag(prompt)
        return response["generated_answer"], response["tokens_count"]
    
class FusionRAGSystem(LLMSystem):
    def __init__(self, system_name: str):
        super().__init__(system_name)

    def query(self, prompt: str) -> Tuple[str, int]:
        # Simulate response generation with Fusion-RAG
        return generate_response_fusion_rag(prompt)
    
class CRAGRAGSystem(LLMSystem):
    def __init__(self, system_name: str):
        super().__init__(system_name)

    def query(self, prompt: str) -> Tuple[str, int]:
        # Simulate response generation with CRAG-RAG
        return generate_response_corrective_rag(prompt)
    
class RerankerRAGSystem(LLMSystem):
    def __init__(self, system_name: str):
        super().__init__(system_name)

    def query(self, prompt: str) -> Tuple[str, int]:
        # Simulate response generation with Reranker-RAG
        response = generate_response_reranker_rag(prompt)  # Assuming similar to CRAG for now
        return response["generated_answer"], response["tokens_count"]
    
class HyDERAGSystem(LLMSystem):
    def __init__(self, system_name: str):
        super().__init__(system_name)

    def query(self, prompt: str) -> Tuple[str, int]:
        # Simulate response generation with HyDE-RAG
        return generate_response_hyde_rag(prompt)  # Assuming similar to NoRAG for now