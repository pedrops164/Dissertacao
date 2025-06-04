from config import config
from openai import OpenAI
from typing import Tuple
from prompts import base_system_prompt
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionMessageParam

nebius_api_key = config.get("nebius_api_key")
base_llm_model = config.get("LLM_MODEL")

openai_client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=nebius_api_key
)

def extract_answer_from_model_response(llm_output: str) -> str:
    """
    Extracts the answer from an LLM response that might contain <think>...</think> blocks.
    """
    think_open_tag = "<think>"
    think_close_tag = "</think>"
    
    llm_output_stripped = llm_output.strip() # Remove leading/trailing whitespace first

    if llm_output_stripped.startswith(think_open_tag):
        # Find the position of the closing </think> tag
        close_tag_start_index = llm_output_stripped.find(think_close_tag)
        
        if close_tag_start_index != -1:
            # Calculate where the actual answer begins
            answer_start_index = close_tag_start_index + len(think_close_tag)
            return llm_output_stripped[answer_start_index:].strip()
        else:
            # It starts with <think> but no </think> was found.
            # This indicates a malformed response.
            raise ValueError("Malformed LLM response: missing closing </think> tag.")
    else:
        # Does not start with <think>, so the whole string is the answer
        return llm_output_stripped

# --- Helper for LLM Assessment Call ---
def call_llm_assessment(prompt=None, max_tokens=1000, temperature=0.0, generation_prompt=None) -> Tuple[str, int]:
    """Makes a focused LLM call for assessment tasks."""
    assert prompt is not None or generation_prompt is not None, "Either prompt or generation_prompt must be provided."
    try:
        if generation_prompt:
            messages = generation_prompt
        else:
            messages = [ChatCompletionUserMessageParam(role="user", content=prompt)]
        response = openai_client.chat.completions.create(
            model=base_llm_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            stop=None,
        )
    except Exception as e:
        print(f"Error during LLM call: {e}")
        return "", 0
    
    raw_llm_output = response.choices[0].message.content
    extracted_answer = extract_answer_from_model_response(raw_llm_output)
    tokens_used = response.usage.total_tokens
    return extracted_answer, tokens_used
    
def query_llm_with_context(query: str, context: str = "", max_tokens: int = 1000, temperature: float = 0.0) -> Tuple[str, int]:
    """Queries the LLM with a given query and optional context."""

    # The system prompt is crucial for guiding behavior.
    # It should instruct the LLM to answer directly and not comment on context.
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=base_system_prompt)
    ]

    # Combine context and query into a single user message
    if context:
        user_content = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
    else:
        user_content = query
    
    messages.append(ChatCompletionUserMessageParam(role="user", content=user_content))
    
    response = openai_client.chat.completions.create(
        model=base_llm_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=1,
        stop=None,
    )
    
    raw_llm_output = response.choices[0].message.content
    extracted_answer = extract_answer_from_model_response(raw_llm_output)
    tokens_used = response.usage.total_tokens
    return extracted_answer, tokens_used
