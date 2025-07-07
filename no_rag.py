# This is the implementation of the base LLM, without any RAG.
import time
from typing import Tuple
from llm_system import LLMSystem
from llm_client import NebiusLLMClient

class NoRAGSystem(LLMSystem):
    def __init__(self, system_name: str, llm_client: NebiusLLMClient):
        super().__init__(system_name, llm_client)

    def query(self, prompt: str, formatted_prompt: str) -> Tuple[str, dict]:
        """
        This function takes a query as input and returns a response from the base LLM as well as useful metrics about the run.
        It uses the OpenAI API to get the response.
        """
        #print(f"\n--- Starting No-RAG Process for Query: '{prompt}' ---")
        start_time = time.time()

        response, token_count = self.llm_client.call_llm_assessment(
            prompt=formatted_prompt,
        )

        end_time = time.time()
        metrics = {
            "token_count": token_count,
            "latency": end_time - start_time,
        }
        #print(f"\n--- No-RAG Process Completed in {end_time - start_time:.2f} seconds ---")
        return response, metrics
