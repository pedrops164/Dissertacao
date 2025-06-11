# This is the implementation of the base LLM, without any RAG.
from llm_client import call_llm_assessment
import time
from typing import Tuple

def generate_response_no_rag(query: str) -> Tuple[str, int]:
    """
    This function takes a query as input and returns a response from the base LLM.
    It uses the OpenAI API to get the response.
    """
    print(f"\n--- Starting No-RAG Process for Query: '{query}' ---")
    start_time = time.time()

    response, token_count = call_llm_assessment(
        temperature=0.1,
        prompt=query,
    )

    end_time = time.time()
    print(f"\n--- No-RAG Process Completed in {end_time - start_time:.2f} seconds ---")
    return response, token_count
