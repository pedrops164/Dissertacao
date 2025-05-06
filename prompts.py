from config import config

lang = config.get("prompts_lang", "en")

crag_assessment_prompt_en = \
"""Evaluate the relevance and sufficiency of the provided 'Retrieved Context' for answering the 'Query'. Consider if the context directly addresses the query and if the query might require more up-to-date information than the context likely provides (suggesting web search).

Query: "{query}"

Retrieved Context:
"{context_snippet}"

Based on your evaluation, choose ONE of the following options:
1: The context is largely irrelevant, clearly insufficient, or the query demands very recent information unlikely to be in the context. Web search is required, and the retrieved context should likely be ignored.
2: The context is relevant but might be incomplete, not fully detailed, or potentially outdated for the query. Supplementing with web search is advisable. Use both the retrieved context and web search results.
3: The context is highly relevant, comprehensive, and directly addresses the query well. It seems sufficient on its own. Web search is likely unnecessary. Use only the retrieved context.

Answer ONLY with the number: 1, 2, or 3."""

crag_rewrite_prompt_en = \
"""Rewrite the following user query to be optimized for a web search engine like Google or Bing. Focus on extracting key terms, removing conversational filler, and making it concise and fact-seeking. If the query implies needing recent information, ensure the rewritten query reflects that.

Original Query: "{query}"

Output ONLY the rewritten web search query."""

def get_crag_assessment_prompt(query: str, context_snippet: str) -> str:
    """
    Get the assessment prompt in English
    """
    if lang != "en":
        raise ValueError("Unsupported language. Only English is supported.")
    return crag_assessment_prompt_en.format(query=query, context_snippet=context_snippet)

def get_crag_rewrite_prompt(query: str) -> str:
    """
    Get the rewrite prompt in English
    """
    if lang != "en":
        raise ValueError("Unsupported language. Only English is supported.")
    return crag_rewrite_prompt_en.format(query=query)



""" SELF RAG PROMPTS """

self_rag_retrieval_prompt = \
"""Does the following query likely require searching external documents for a factual and comprehensive answer, or can it be answered reliably from general knowledge?
Query: "{query}"
Answer ONLY with YES or NO."""

self_rag_critique_prompt = \
"""Evaluate if the following document passage is relevant and helpful for answering the query. Consider if it directly addresses the query or provides useful background.
Query: "{query}"
Passage: "{doc_text}"
Answer ONLY with RELEVANT or IRRELEVANT."""

def get_self_rag_retrieval_prompt(query: str) -> str:
    """
    Get the retrieval prompt in English
    """
    if lang != "en":
        raise ValueError("Unsupported language. Only English is supported.")
    return self_rag_retrieval_prompt.format(query=query)

def get_self_rag_critique_prompt(query: str, doc_text: str) -> str:
    """
    Get the critique prompt in English
    """
    if lang != "en":
        raise ValueError("Unsupported language. Only English is supported.")
    return self_rag_critique_prompt.format(query=query, doc_text=doc_text[:1000])

def get_self_rag_generation_prompt_message(query: str, filtered_context: str):
    """
    Get the generation prompt message in English
    """
    if lang != "en":
        raise ValueError("Unsupported language. Only English is supported.")
    
    if filtered_context:
        system_prompt = "You are a helpful AI assistant. Answer the user's question based *only* on the provided relevant context. Be factual and concise."
        user_prompt = \
"""Relevant Context:
{filtered_context}

Question: {query}

Based *only* on the relevant context provided, answer the question."""
    else:
        system_prompt = "You are a helpful AI assistant. Answer the user's question directly using your general knowledge."
        user_prompt = f"Question: {query}"

    generation_prompt_message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return generation_prompt_message

self_rag_critique_answer_prompt = \
"""Evaluate the following generated answer based SOLELY on the provided query and context (if any).
Is the answer factually consistent with the context?
Is the answer relevant to the original query?

Query: "{query}"
Context Provided: "{filtered_context}"
Generated Answer: "{generated_answer}"

Answer ONLY with one word: SUPPORTED, CONTRADICTORY, or NOT_SUPPORTED (if answer is irrelevant or not verifiable from context). If no context was provided, answer NOT_APPLICABLE."""

def get_self_rag_critique_answer_prompt(query: str, filtered_context: str, generated_answer: str) -> str:
    """
    Get the critique answer prompt in English
    """
    if lang != "en":
        raise ValueError("Unsupported language. Only English is supported.")
    return self_rag_critique_answer_prompt.format(query=query, filtered_context=filtered_context[:1000], generated_answer=generated_answer)