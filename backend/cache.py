import numpy as np
from collections import OrderedDict

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

class SemanticCache:
    def __init__(self, capacity: int = 500, threshold: float = 0.92):
        self.capacity = capacity
        self.threshold = threshold
        # Structure: {(slug, id): { 'embedding': np.array, 'response': str, 'mode': str, 'sources': list }}
        self.cache = OrderedDict()
        self._id_counter = 0

    def get(self, university_slug: str, query_embedding: list):
        query_vec = np.array(query_embedding)
        best_match_key = None
        best_sim = -1.0
        
        # Linear search for best above threshold
        for key, data in self.cache.items():
            if key[0] != university_slug:
                continue
            
            sim = cosine_similarity(query_vec, data['embedding'])
            if sim > best_sim and sim >= self.threshold:
                best_sim = sim
                best_match_key = key
                
        if best_match_key:
            # Move to end (LRU)
            val = self.cache.pop(best_match_key)
            self.cache[best_match_key] = val
            return val
            
        return None

    def set(self, university_slug: str, query_embedding: list, response: str, mode: str, sources: list):
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False) # remove first inserted (least recently used)
            
        key = (university_slug, self._id_counter)
        self._id_counter += 1
        
        self.cache[key] = {
            'embedding': np.array(query_embedding),
            'response': response,
            'mode': mode,
            'sources': sources
        }

# Global singleton cache instance
semantic_cache = SemanticCache()
