from abc import ABC, abstractmethod
from no_rag import generate_response_no_rag
from self_rag import generate_response_self_rag
from fusion_rag import generate_response_fusion_rag
from crag_rag import generate_response_corrective_rag

class LLMSystem(ABC):
    def __init__(self, system_name: str):
        self.system_name = system_name

    @abstractmethod
    def query(self, prompt: str) -> str:
        pass

class NoRAGSystem(LLMSystem):
    def __init__(self, system_name: str):
        super().__init__(system_name)

    def query(self, prompt: str) -> str:
        # Simulate response generation without RAG
        return generate_response_no_rag(prompt)
    
class SelfRAGSystem(LLMSystem):
    def __init__(self, system_name: str):
        super().__init__(system_name)

    def query(self, prompt: str) -> str:
        # Simulate response generation with Self-RAG
        response =  generate_response_self_rag(prompt)
        return response["generated_answer"], response["tokens_count"]
    
class FusionRAGSystem(LLMSystem):
    def __init__(self, system_name: str):
        super().__init__(system_name)

    def query(self, prompt: str) -> str:
        # Simulate response generation with Fusion-RAG
        return generate_response_fusion_rag(prompt)
    
class CRAGRAGSystem(LLMSystem):
    def __init__(self, system_name: str):
        super().__init__(system_name)

    def query(self, prompt: str) -> str:
        # Simulate response generation with CRAG-RAG
        return generate_response_corrective_rag(prompt)