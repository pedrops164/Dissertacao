from llm_client import call_llm_assessment

def generate_response_rag(query, context=""):
    """
    Generate a response based on the query using RAG.
    Retrieves context from the vector database before calling the LLM.

    Args:
        query (str): User query

    Returns:
        str: Generated response
    """
    # make prompts in english
    system_prompt = """You are an AI assistant. Answer the user's question based on the provided context.
    If the context contains irrelevant information to answer the question, ignore it and use your knowledge."""
    # Format the user prompt with the context and query
    user_prompt = f"""Context:
    {context}
    Question: {query}
    Please answer the question."""
    
    response, token_count = call_llm_assessment(
        temperature=0.1,
        generation_prompt=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # Return the generated response
    return response, token_count

# Example usage
if __name__ == "__main__":
    query = "What is the capital of France?"
    context = "Paris is the capital and most populous city of France."
    
    response, tokens = generate_response_rag(query, context)
    print(f"Response: {response}\nTokens used: {tokens}")