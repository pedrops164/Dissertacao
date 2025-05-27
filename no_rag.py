# This is the implementation of the base LLM, without any RAG.
from llm_client import call_llm_assessment
import time

def generate_response_no_rag(query):
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

if __name__ == "__main__":
    # Example usage
    query = "Quem escreveu 'Os Lusíadas'?"
    response = generate_response_no_rag(query)
    print(f"Response: {response}")