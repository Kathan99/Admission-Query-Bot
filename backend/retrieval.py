import os
import asyncio
from typing import Optional
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils import embedding_functions

# To allow relative imports if run as a script or module
try:
    from backend.config import settings
    from backend.generation import GenerationManager
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.config import settings
    from backend.generation import GenerationManager

class RetrievalManager:
    _chroma_client = None
    _embedding_function = None
    _reranker = None
    
    # In-memory cache for BM25 models: { university_slug: (BM25Okapi, list_of_docs, list_of_metadatas) }
    _bm25_cache = {}

    @classmethod
    def _init_models(cls):
        if not cls._chroma_client:
            cls._chroma_client = chromadb.PersistentClient(path=settings.chroma_db_dir)
        if not cls._embedding_function:
            cls._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model
            )
        if getattr(cls, '_reranker_failed', None) is None:
            cls._reranker_failed = False
            
        if not cls._reranker and not cls._reranker_failed:
            try:
                cls._reranker = CrossEncoder(settings.reranker_model, max_length=512)
            except Exception as e:
                print(f"Warning: Reranker unavailable, falling back to similarity scores. Error: {e}")
                cls._reranker_failed = True

    @classmethod
    def _get_or_build_bm25(cls, university_slug: str, collection):
        if university_slug in cls._bm25_cache:
            return cls._bm25_cache[university_slug]
            
        # Fetch all documents to build BM25 (lazy loading)
        # In a real massive production system, this would be updated on ingestion natively
        results = collection.get(include=['documents', 'metadatas'])
        if not results or not results['documents']:
            cls._bm25_cache[university_slug] = (None, [], [])
            return cls._bm25_cache[university_slug]
            
        docs = results['documents']
        metadatas = results['metadatas']
        
        tokenized_corpus = [doc.lower().split() for doc in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        
        cls._bm25_cache[university_slug] = (bm25, docs, metadatas)
        return cls._bm25_cache[university_slug]

    @classmethod
    async def retrieve(cls, query: str, university_slug: Optional[str] = None):
        cls._init_models()
        
        collections_to_query = []
        if university_slug and university_slug != "global_search":
            try:
                collections_to_query.append(cls._chroma_client.get_collection(
                    name=f"uni_{university_slug}", 
                    embedding_function=cls._embedding_function
                ))
            except Exception:
                pass
        else:
            try:
                # get all collections
                for col in cls._chroma_client.list_collections():
                    if col.name.startswith("uni_"):
                        collections_to_query.append(cls._chroma_client.get_collection(
                            name=col.name, 
                            embedding_function=cls._embedding_function
                        ))
            except Exception:
                pass

        if not collections_to_query:
            return []

        # Step 1: HyDE query rewriting
        hyde_query = GenerationManager.generate_hyde_response(query)
        hybrid_search_query = f"{query} {hyde_query}"

        all_dense_docs = []
        all_dense_metas = []
        all_bm25_docs = []
        all_bm25_metas = []
        global_all_docs = []
        global_all_metas = []

        async def query_single_collection(col):
            def run_bm25_search():
                bm_slug = col.name.replace("uni_", "")
                bm25_model, all_d, all_m = cls._get_or_build_bm25(bm_slug, col)
                if not bm25_model:
                    return [], [], all_d, all_m
                tokenized_query = query.lower().split()
                bm25_scores = bm25_model.get_scores(tokenized_query)
                top_10_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
                b_docs = [all_d[i] for i in top_10_indices if bm25_scores[i] > 0]
                b_metas = [all_m[i] for i in top_10_indices if bm25_scores[i] > 0]
                return b_docs, b_metas, all_d, all_m

            dense_task = asyncio.to_thread(
                col.query,
                query_texts=[hybrid_search_query],
                n_results=10,
                include=['documents', 'metadatas']
            )
            bm25_task = asyncio.to_thread(run_bm25_search)
            return await asyncio.gather(dense_task, bm25_task)

        # Gather across all collections concurrently
        tasks = [query_single_collection(col) for col in collections_to_query]
        results = await asyncio.gather(*tasks)

        for dense_results, (b_docs, b_metas, all_d, all_m) in results:
            if dense_results['documents'] and dense_results['documents'][0]:
                all_dense_docs.extend(dense_results['documents'][0])
                all_dense_metas.extend(dense_results['metadatas'][0])
            all_bm25_docs.extend(b_docs)
            all_bm25_metas.extend(b_metas)
            global_all_docs.extend(all_d)
            global_all_metas.extend(all_m)
        
        # Step 4: Reciprocal Rank Fusion
        fused_scores = {}
        for r, d in enumerate(all_dense_docs):
            if d not in fused_scores:
                fused_scores[d] = 0.0
            fused_scores[d] += 1 / (60 + r)
            
        for r, d in enumerate(all_bm25_docs):
            if d not in fused_scores:
                fused_scores[d] = 0.0
            fused_scores[d] += 1 / (60 + r)
            
        # Sort fused candidates
        fused_candidates = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        # Limit to 10 candidates total for reranking
        fused_candidates = fused_candidates[:10]
        
        if not fused_candidates:
            return []
            
        # Step 5: Cross-encoder Reranking
        if getattr(cls, '_reranker_failed', False) or not cls._reranker:
            # Fallback: Just use top 5 from fused_candidates
            # Assign dummy scores based on their fusion rank
            doc_score_pairs = [(doc, float(1.0 - (i * 0.1))) for i, doc in enumerate(fused_candidates[:5])]
            top_5_pairs = doc_score_pairs
        else:
            try:
                cross_inp = [[query, doc] for doc in fused_candidates]
                rerank_scores = cls._reranker.predict(cross_inp)
                
                # Combine candidate docs with scores and sort
                doc_score_pairs = list(zip(fused_candidates, rerank_scores))
                doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Select top 5
                top_5_pairs = doc_score_pairs[:5]
            except Exception as e:
                print(f"Warning: Reranker prediction failed, falling back to similarity scores. Error: {e}")
                doc_score_pairs = [(doc, float(1.0 - (i * 0.1))) for i, doc in enumerate(fused_candidates[:5])]
                top_5_pairs = doc_score_pairs
        
        # Get metadata for the top 5
        final_results = []
        
        def _find_meta(doc_text):
            # find first occurrence of doc in dense or all
            for i, d in enumerate(all_dense_docs):
                if d == doc_text: return all_dense_metas[i]
            for i, d in enumerate(global_all_docs):
                if d == doc_text: return global_all_metas[i]
            return {}

        for doc_text, score in top_5_pairs:
            meta = _find_meta(doc_text)
            final_results.append({
                "text": doc_text,
                "score": float(score),
                "metadata": meta
            })
            
        return final_results
