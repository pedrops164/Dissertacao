
from tavily import TavilyClient # Import Tavily client

import os
from config import config

# --- Configuration ---
TAVILY_API_KEY = config.get("tavily_api_key") # Get Tavily API key from config

# Initialize Tavily Client (only if API key exists)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

# Max results to fetch from Tavily
WEB_SEARCH_MAX_RESULTS = 3
# Max characters per result to include in context (to keep it concise)
WEB_SEARCH_MAX_CHARS_PER_RESULT = 1500
# Search depth for Tavily ('basic' or 'advanced')
TAVILY_SEARCH_DEPTH = "basic"

def search_tavily(query, search_depth=TAVILY_SEARCH_DEPTH, max_results=WEB_SEARCH_MAX_RESULTS):
    """
    Perform a web search using the Tavily API based on the query.
    """
    try:
        response = tavily_client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=False, # Often less useful than individual results for RAG context
            include_raw_content=False # Usually don't need full raw HTML/text
        )

        # Process the results
        results = response.get('results', [])
        if not results:
            print("  > Tavily search returned no results.")
            return ""

        # Extract content/snippets and concatenate
        contexts = []
        for i, res in enumerate(results):
            content = res.get('content', '')
            # Limit character count per result to avoid excessive context length
            if content:
                 contexts.append(f"{content[:WEB_SEARCH_MAX_CHARS_PER_RESULT]}...")

        if not contexts:
            print("  > Tavily results found, but no usable content/snippets extracted.")
            return ""

        final_context = "\n\n---\n\n".join(contexts)
        print(f"  > Retrieved {len(contexts)} snippets from Tavily. Total length: {len(final_context)} chars.")
        return final_context

    except Exception as e:
        print(f"  > Error during Tavily API call for '{query}': {e}")
        return "" # Return empty string on error