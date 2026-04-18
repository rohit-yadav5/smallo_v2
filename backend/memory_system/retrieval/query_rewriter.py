def rewrite_query_for_retrieval(query: str) -> str:
    """Expand query into key concepts for better FAISS semantic matching.

    Returns the original query unchanged if the LLM call fails or returns
    an empty result — so the caller always gets something usable.
    """
    try:
        result = _call_rewriter_llm(query)
        return result.strip() if result and len(result.strip()) > 5 else query
    except Exception:
        return query


def _call_rewriter_llm(query: str) -> str:
    from llm import ask_llm
    prompt = (
        "Rewrite this search query as space-separated key concepts and related terms "
        "for semantic vector search. Output only the terms, no explanation, no punctuation.\n\n"
        f"Query: {query}\n\nExpanded terms:"
    )
    return ask_llm(prompt).strip()
