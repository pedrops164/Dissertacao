import os
from openai import OpenAI
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

openai_client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("NEBIUS_API_KEY")  # Retrieve the API key from environment variables
)

# --- Helper for LLM Assessment Call ---
def call_llm_assessment(prompt=None, max_tokens=5, temperature=0.0, generation_prompt=None) -> str:
    """Makes a focused LLM call for assessment tasks."""
    try:
        response = openai_client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct", # Or your chosen assessment model
            #messages=[{"role": "user", "content": prompt}],
            messages=generation_prompt if generation_prompt else [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            stop=None,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling LLM for assessment: {e}")
        return None
    