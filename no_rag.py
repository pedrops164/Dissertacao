# This is the implementation of the base LLM, without any RAG.
from llm_client import openai_client
import time

def generate_response_no_rag(query):
    """
    This function takes a query as input and returns a response from the base LLM.
    It uses the OpenAI API to get the response.
    """
    print(f"\n--- Starting No-RAG Process for Query: '{query}' ---")
    start_time = time.time()

    # Get the response from the OpenAI API
    response = openai_client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[{"role": "user", "content": query}],
        temperature=0,
        max_tokens=1000,
        n=1,
        stop=None,
    )

    # Extract the text from the response
    text = response.choices[0].message.content

    end_time = time.time()
    print(f"\n--- No-RAG Process Completed in {end_time - start_time:.2f} seconds ---")
    return text

if __name__ == "__main__":
    # Example usage
    query = "Quem escreveu 'Os Lusíadas'?"
    response = generate_response_no_rag(query)
    print(f"Response: {response}")