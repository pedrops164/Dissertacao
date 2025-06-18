from abc import ABC, abstractmethod
from llm_client import NebiusLLMClient
from typing import Tuple

class LLMSystem(ABC):
    def __init__(self, system_name: str, llm_client: NebiusLLMClient):
        self.system_name = system_name
        self.llm_client = llm_client

    @abstractmethod
    def query(self, prompt: str, formatted_prompt: str) -> Tuple[str, dict]:
        pass