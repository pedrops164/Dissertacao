from config import config

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
    Get the assessment prompt
    """
    return crag_assessment_prompt_en.format(query=query, context_snippet=context_snippet)

def get_crag_rewrite_prompt(query: str) -> str:
    """
    Get the rewrite prompt
    """
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
    Get the retrieval prompt
    """
    return self_rag_retrieval_prompt.format(query=query)

def get_self_rag_critique_prompt(query: str, doc_text: str) -> str:
    """
    Get the critique prompt
    """
    return self_rag_critique_prompt.format(query=query, doc_text=doc_text[:1000])

def get_self_rag_generation_prompt_message(query: str, filtered_context: str):
    """
    Get the generation prompt message
    """
    
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
    Get the critique answer prompt
    """
    return self_rag_critique_answer_prompt.format(query=query, filtered_context=filtered_context[:1000], generated_answer=generated_answer)

""" HYDE RAG PROMPTS """

def get_hyde_rag_prompt(user_query: str) -> str:
    """
    Generates a system prompt for an LLM to create a hypothetical document for HyDE RAG.

    This function takes a user's query and embeds it into a larger prompt
    that instructs the LLM to generate a hypothetical research paper abstract
    in a specific format (Title and Abstract). This generated document is designed
    to resemble the chunks in a medical vector database.

    Args:
        user_query: The original question from the user.

    Returns:
        A formatted string containing the full system prompt to be sent to the LLM.
    """

    # The core prompt template with instructions, a one-shot example, and a placeholder for the user query.
    # Using a raw f-string (rf"""...""") is good practice to handle backslashes if any exist.
    hyde_prompt = rf"""
You are an expert medical research assistant. Your task is to take a user's question and generate a hypothetical research paper abstract that provides a detailed answer. This generated document will be used to find similar, real documents in a database.

The generated document **must** follow this exact format:

**Title:** \[A formal, scientific title for a hypothetical research paper that answers the query]
**Abstract:** \[A detailed abstract for the hypothetical paper. It should describe the background, methods, results, and conclusion of a plausible study that answers the user's question. Use precise medical terminology and maintain a scientific tone.]

---

### **Example**

**User Query:** "What is a treatment plan for spontaneous leukemia in AKR mice using immunotherapy and chemoradiotherapy?"

**Generated Document:**
**Title:** Graft versus leukemia. VI. Adoptive immunotherapy in combination with chemoradiotherapy for spontaneous leukemia-lymphoma in AKR mice.
**Abstract:** A three-step treatment plan incorporating adoptive immunotherapy and chemoradiotherapy was used to treat AKR (H-2k) mice bearing spontaneous leukemia-lymphoma (SLL). 1) Leukemic mice were treated with chemoradiotherapy for immunosuppression and leukemia cytoreduction. 2) To introduce a graft-versus-leukemia reaction against residual malignant cells, the immunosuppressed AKR mice were given immunocompetent cells from H-2 mismatched DBA/2 (H-2d) donors. 3) To "rescue" the AKR hosts from incipient graft-versus-host disease, the mismatched DBA/2 cells were killed with combination chemotherapy, and cells from allogeneic H-2 matched RF (H-2k) donors were administered to restore hematopoiesis. Leukemic AKR mice thus treated had significant prolongation of their median survival time and a higher 60-day survival rate post treatment than did untreated controls, chemoradiotherapy controls, or control mice that received chemoradiotherapy plus cells from syngeneic donors. Therefore, adoptive immunotherapy may be useful as an adjunct to conventional therapy for treatment of SLL in AKR mice.

---

Now, please generate a hypothetical document for the following user query. Create a plausible, detailed, and scientifically-toned Title and Abstract. Don't include the answer to the question.

**User Query:** "{user_query}"
"""
    return hyde_prompt



base_system_prompt = "You are a helpful assistant. Answer the user's question directly. If context is provided, use it to inform your answer. DO NOT MENTION THE CONTEXT OR ITS RELEVANCE IN YOUR RESPONSE."

llm_judge_prompt = \
"""Evaluate the following response to a query. Rate each aspect on a scale of 1-10.

Query: {query}

Ground Truth: {ground_truth}

Response to Evaluate:
{response}

Please evaluate the response on the following criteria:
1. Factual Correctness (1-10): Is the information in the response factually accurate according to the retrieved context?
2. Answer Relevance (1-10): How relevant is the response to the query?
3. Hallucination (1-10): Does the response contain information not supported by the retrieved context? (10 = no hallucination, 1 = completely hallucinated)
4. Completeness (1-10): Does the response address all aspects of the query?
5. Coherence (1-10): Is the response well-structured, logical, and easy to understand?

Provide your ratings in the following JSON format:
```json
{{
  "factual_correctness": 0,
  "answer_relevance": 0,
  "hallucination_score": 0,
  "completeness": 0,
  "coherence": 0
}}
```
"""
def get_llm_judge_prompt(query: str, ground_truth: str, response: str) -> str:
    """
    Get the LLM judge prompt
    """
    return llm_judge_prompt.format(query=query, ground_truth=ground_truth, response=response)