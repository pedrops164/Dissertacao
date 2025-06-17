# This is the implementation of the base LLM, without any RAG.
from llm_client import call_llm_assessment
import time
from typing import Tuple

def generate_response_no_rag(query: str, formatted_query: str) -> Tuple[str, dict]:
    """
    This function takes a query as input and returns a response from the base LLM as well as useful metrics about the run.
    It uses the OpenAI API to get the response.
    """
    print(f"\n--- Starting No-RAG Process for Query: '{query}' ---")
    start_time = time.time()

    response, token_count = call_llm_assessment(
        prompt=formatted_query,
    )

    end_time = time.time()
    metrics = {
        "token_count": token_count,
        "latency": end_time - start_time,
    }
    print(f"\n--- No-RAG Process Completed in {end_time - start_time:.2f} seconds ---")
    return response, metrics
