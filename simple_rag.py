from llm_client import openai_client

def generate_response_rag(query, context=""):
    """
    Generate a response based on the query using RAG.
    Retrieves context from the vector database before calling the LLM.

    Args:
        query (str): User query

    Returns:
        str: Generated response
    """
    system_prompt = """És um assistente de IA. Responde à pergunta do utilizador com base no contexto fornecido.
    Se o contexto contiver informações irrelevantes para responder à pergunta, ignora-o e utiliza o teu conhecimento."""

    # Format the user prompt with the context and query
    user_prompt = f"""Contexto:
    {context}

    Pergunta: {query}

    Por favor, responde à pergunta."""

    # Generate the response using the OpenAI API
    response = openai_client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # Specify the model to use
        messages=[
            {"role": "system", "content": system_prompt},  # System message to guide the assistant
            {"role": "user", "content": user_prompt}  # User message with context and query
        ],
        temperature=0.1  # Set the temperature for response generation
    )
    
    # Return the generated response
    return response.choices[0].message.content
