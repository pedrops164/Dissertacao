from config import config
from openai import OpenAI
from typing import Tuple
from prompts import base_system_prompt

nebius_api_key = config.get("nebius_api_key")

openai_client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=nebius_api_key
)

# --- Helper for LLM Assessment Call ---
def call_llm_assessment(prompt=None, max_tokens=1000, temperature=0.0, generation_prompt=None) -> Tuple[str, int]:
    """Makes a focused LLM call for assessment tasks."""
    try:
        response = openai_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            messages=generation_prompt if generation_prompt else [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            stop=None,
        )
    except Exception as e:
        print(f"Error during LLM call: {e}")
        return "", 0
    return response.choices[0].message.content.strip(), response.usage.total_tokens
    
def query_llm_with_context(query: str, context: str = "", max_tokens: int = 1000, temperature: float = 0) -> Tuple[str, int]:
    """Queries the LLM with a given query and optional context."""
    messages = [
        {"role": "system", "content": base_system_prompt},
        {"role": "user", "content": query}
    ]
    
    #if context:
    #    messages.append({"role": "assistant", "content": context})
    
    response = openai_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=1,
        stop=None,
    )
    
    return response.choices[0].message.content.strip(), response.usage.total_tokens