import re

class KnowledgeRouter:
    MODE_RAG_ONLY = "RAG_ONLY"
    MODE_LLM_ONLY = "LLM_ONLY"
    MODE_BLEND = "BLEND"

    # Keywords that signal the question asks for very specific document facts.
    # For these we still prefer RAG_ONLY so the LLM is pushed to cite the doc first,
    # but the new prompts will fall back to general knowledge if the doc doesn't have it.
    RAG_PRIORITY_KEYWORDS = [
        "seat", "seats", "fee", "fees", "cutoff", "cut-off", "rank",
        "deadline", "scholarship", "hostel", "mess", "syllabus", "intake",
        "placement", "placement record", "stipend",
    ]

    @staticmethod
    def contains_specific_facts_query(query: str) -> bool:
        query_lower = query.lower()
        if any(kw in query_lower for kw in KnowledgeRouter.RAG_PRIORITY_KEYWORDS):
            return True
        if re.search(r'\bhow many\b|\bamount\b|\bexact\b', query_lower):
            return True
        return False

    @staticmethod
    def route(query: str, top_rerank_score: float, has_docs: bool = True) -> str:
        """
        Determines the routing mode.

        - No docs available  → LLM_ONLY (Gemini answers from its own knowledge)
        - Specific-fact query (fees, seats, cutoffs…) + docs → RAG_ONLY
          (prompt instructs LLM to cite doc first, fall back to knowledge if not found)
        - Everything else    → BLEND (docs + general knowledge, always a full answer)
        """
        if not has_docs:
            # No document chunks retrieved — Gemini answers from its own knowledge
            return KnowledgeRouter.MODE_LLM_ONLY

        if KnowledgeRouter.contains_specific_facts_query(query):
            # Push LLM to cite the document for hard facts, but still allow knowledge fallback
            return KnowledgeRouter.MODE_RAG_ONLY

        # Default: blend document context with Gemini's broad education knowledge
        return KnowledgeRouter.MODE_BLEND
