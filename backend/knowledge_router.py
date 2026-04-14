import re

class KnowledgeRouter:
    MODE_RAG_ONLY = "RAG_ONLY"
    MODE_LLM_ONLY = "LLM_ONLY"
    MODE_BLEND = "BLEND"
    
    RAG_ONLY_KEYWORDS = [
        "seat", "seats", "fee", "fees", "cutoff", "cut-off", "rank", 
        "deadline", "scholarship", "hostel", "mess", "syllabus"
    ]
    
    @staticmethod
    def contains_specific_facts_query(query: str) -> bool:
        query_lower = query.lower()
        if any(kw in query_lower for kw in KnowledgeRouter.RAG_ONLY_KEYWORDS):
            return True
        # Check for numeric references like "how many", "amount", percentages, dates
        if re.search(r'\bhow many\b|\bamount\b|\bexact\b', query_lower):
            return True
        return False

    @staticmethod
    def route(query: str, top_rerank_score: float, has_docs: bool = True) -> str:
        """
        Determines the routing mode based on the max score from provided documents
        and query properties.
        """
        if not has_docs:
            return KnowledgeRouter.MODE_LLM_ONLY
            
        needs_specific_facts = KnowledgeRouter.contains_specific_facts_query(query)

        # We used to check `top_rerank_score < 0.40` for LLM_ONLY, but CrossEncoders output raw logits 
        # (often negative, like -8.0 to +10.0), which caused us to inadvertently discard perfectly good context!
        
        # If there are documents, we should almost always default to BLEND or RAG_ONLY, and let the LLM
        # decide if the provided context actually contains the answer.

        if needs_specific_facts:
            return KnowledgeRouter.MODE_RAG_ONLY

        return KnowledgeRouter.MODE_BLEND
