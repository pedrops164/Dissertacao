from config import config
from openai import OpenAI
from typing import Tuple
from prompts import base_system_prompt
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionMessageParam

judge_llm_model = config.get("JUDGE_LLM_MODEL")

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

class NebiusLLMClient():
    def __init__(self, base_llm: str):
        nebius_api_key = config.get("nebius_api_key")
        self.openai_client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=nebius_api_key
        )
        self.base_llm = base_llm

    # --- Helper for LLM Assessment Call ---
    def call_llm_assessment(self, prompt, temperature=0.0, use_judge_model=False) -> Tuple[str, int]:
        """Makes a focused LLM call for assessment tasks."""
        try:
            model = judge_llm_model if use_judge_model else self.base_llm
            messages = [ChatCompletionUserMessageParam(role="user", content=prompt)]
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
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
        
    def query_llm_with_context(self, query: str, context: str = "", temperature: float = 0.0) -> Tuple[str, int]:
        """Queries the LLM with a given query and optional context."""

        # The system prompt is crucial for guiding behavior.
        # It should instruct the LLM to answer directly and not comment on context.
        #messages: list[ChatCompletionMessageParam] = [
        #    ChatCompletionSystemMessageParam(role="system", content=base_system_prompt)
        #]
        messages: list[ChatCompletionMessageParam] = []

        # Combine context and query into a single user message
        if context:
            user_content = f"Possibly relevant documents:\n{context}\n\n###\
    \nYou are a medical expert. Use the context provided above only if it is clearly relevant to the question. If the context is irrelevant or incomplete, answer based on your own medical knowledge. Always aim to provide an accurate and concise answer.\
    \n\nQuestion: {query}"
        else:
            user_content = query
        
        messages.append(ChatCompletionUserMessageParam(role="user", content=user_content))
        
        response = self.openai_client.chat.completions.create(
            model=self.base_llm,
            messages=messages,
            temperature=temperature,
            n=1,
            stop=None,
        )
        
        raw_llm_output = response.choices[0].message.content
        extracted_answer = extract_answer_from_model_response(raw_llm_output)
        tokens_used = response.usage.total_tokens
        return extracted_answer, tokens_used
