import os
import asyncio
import hashlib
import json
from typing import Optional, Dict, Any, List, Tuple

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils import embedding_functions

# Optional Redis for caching HyDE
import redis

# To allow relative imports if run as a script or module
try:
    from backend.config import settings, load_universities
    from backend.generation import GenerationManager
    from backend.knowledge_router import KnowledgeRouter
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.config import settings, load_universities
    from backend.generation import GenerationManager
    from backend.knowledge_router import KnowledgeRouter


class RetrievalManager:
    _chroma_client = None
    _embedding_function = None
    _reranker = None

    _registry_ready = False
    _redis_client = None

    @classmethod
    def _init_models(cls):
        if cls._chroma_client is None:
            try:
                cls._chroma_client = chromadb.PersistentClient(path=settings.chroma_db_dir)
            except Exception as e:
                print(f"Warning: Chroma DB unavailable ({e}); retrieval will return no docs.")
                cls._chroma_client = False  # sentinel: disabled

        if cls._chroma_client is False:
            return

        if not cls._embedding_function:
            cls._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model
            )

        if getattr(cls, "_reranker_failed", None) is None:
            cls._reranker_failed = False

        if not cls._reranker and not cls._reranker_failed:
            try:
                cls._reranker = CrossEncoder(settings.reranker_model, max_length=512)
            except Exception as e:
                print(f"Warning: Reranker unavailable, falling back to fusion rank. Error: {e}")
                cls._reranker_failed = True

        if cls._redis_client is None:
            try:
                redis_host = os.getenv("REDIS_HOST", "localhost")
                redis_port = int(os.getenv("REDIS_PORT", "6379"))
                cls._redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
                cls._redis_client.ping()
            except Exception:
                cls._redis_client = False  # sentinel: disabled

    @classmethod
    def _ensure_registry(cls):
        if settings.single_corpus:
            cls._registry_ready = True
            return
        if cls._registry_ready:
            return
        try:
            registry = cls._chroma_client.get_or_create_collection(
                name="uni_registry",
                embedding_function=cls._embedding_function,
                metadata={"hnsw:space": "cosine"},
            )

            universities = load_universities() or []
            if not universities:
                cls._registry_ready = True
                return

            docs: List[str] = []
            metas: List[Dict[str, Any]] = []
            ids: List[str] = []

            for u in universities:
                slug = u.get("slug")
                if not slug:
                    continue
                name = u.get("name", slug)
                location = u.get("location", "")
                known_for = ", ".join(u.get("known_for", []) or [])
                exams = ", ".join(u.get("entrance_exams", []) or [])
                uni_type = u.get("type", "")

                doc = (
                    f"University: {name}\n"
                    f"Location: {location}\n"
                    f"Type: {uni_type}\n"
                    f"Known for: {known_for}\n"
                    f"Entrance exams: {exams}\n"
                    f"Website: {u.get('website', '')}\n"
                )

                docs.append(doc)
                metas.append({"slug": slug, "name": name, "location": location})
                ids.append(slug)

            if docs:
                # Upsert is idempotent
                registry.upsert(documents=docs, metadatas=metas, ids=ids)

            cls._registry_ready = True
        except Exception as e:
            print(f"Warning: failed to build registry collection: {e}")
            cls._registry_ready = True

    @classmethod
    def _select_global_slugs(cls, query_embedding: List[float]) -> List[str]:
        if settings.global_search_top_n <= 0:
            return []
        try:
            cls._ensure_registry()
            registry = cls._chroma_client.get_collection(
                name="uni_registry",
                embedding_function=cls._embedding_function,
            )
            res = registry.query(
                query_embeddings=[query_embedding],
                n_results=settings.global_search_top_n,
                include=["metadatas"],
            )
            metas = (res.get("metadatas") or [[]])[0]
            slugs = [m.get("slug") for m in metas if isinstance(m, dict) and m.get("slug")]
            # unique, preserve order
            out: List[str] = []
            for s in slugs:
                if s not in out:
                    out.append(s)
            return out
        except Exception:
            return []

    @classmethod
    def _resolve_collections(cls, university_slug: Optional[str], query_embedding: List[float]):
        # Single-corpus mode: always query the one shared collection.
        if settings.single_corpus:
            try:
                return [
                    cls._chroma_client.get_or_create_collection(
                        name=settings.corpus_collection,
                        embedding_function=cls._embedding_function,
                        metadata={"hnsw:space": "cosine"},
                    )
                ]
            except Exception:
                return []

        cols = []
        if university_slug and university_slug != "global_search":
            try:
                cols.append(
                    cls._chroma_client.get_collection(
                        name=f"uni_{university_slug}",
                        embedding_function=cls._embedding_function,
                    )
                )
                return cols
            except Exception:
                return []

        # Global search: first pick top-N relevant universities using the registry.
        selected = cls._select_global_slugs(query_embedding)
        if selected:
            for slug in selected:
                try:
                    cols.append(
                        cls._chroma_client.get_collection(
                            name=f"uni_{slug}",
                            embedding_function=cls._embedding_function,
                        )
                    )
                except Exception:
                    continue
            if cols:
                return cols

        # Fallback: query all uni_* collections
        try:
            for col in cls._chroma_client.list_collections():
                if col.name.startswith("uni_") and col.name != "uni_registry":
                    cols.append(
                        cls._chroma_client.get_collection(
                            name=col.name,
                            embedding_function=cls._embedding_function,
                        )
                    )
        except Exception:
            pass
        return cols

    @staticmethod
    def _should_use_hyde(query: str) -> bool:
        if not settings.enable_hyde:
            return False
        # HyDE is most useful for vague/broad queries; skip for “specific facts” queries.
        try:
            if KnowledgeRouter.contains_specific_facts_query(query):
                return False
        except Exception:
            pass
        # Also skip for very short queries (often better to retrieve literally)
        if len(query.strip().split()) <= 3:
            return False
        return True

    @classmethod
    def _hyde_cache_key(cls, query: str) -> str:
        h = hashlib.sha1(query.strip().encode("utf-8")).hexdigest()
        return f"hyde:v1:{settings.hyde_model}:{h}"

    @classmethod
    def _get_cached_hyde(cls, query: str) -> Optional[str]:
        if not cls._redis_client or cls._redis_client is False:
            return None
        try:
            return cls._redis_client.get(cls._hyde_cache_key(query))
        except Exception:
            return None

    @classmethod
    def _set_cached_hyde(cls, query: str, hyde_text: str):
        if not cls._redis_client or cls._redis_client is False:
            return
        try:
            cls._redis_client.set(cls._hyde_cache_key(query), hyde_text, ex=settings.hyde_cache_ttl_seconds)
        except Exception:
            return

    @classmethod
    async def _get_or_generate_hyde(cls, query: str) -> str:
        cached = cls._get_cached_hyde(query)
        if cached:
            return cached
        hyde = await asyncio.to_thread(GenerationManager.generate_hyde_response, query)
        if hyde:
            cls._set_cached_hyde(query, hyde)
        return hyde

    @classmethod
    async def _dense_query_all(cls, cols, query_embedding: List[float]):
        async def query_one(col):
            return await asyncio.to_thread(
                col.query,
                query_embeddings=[query_embedding],
                n_results=settings.dense_k,
                include=["documents", "metadatas", "distances"],
            )

        return await asyncio.gather(*[query_one(c) for c in cols])

    @classmethod
    async def retrieve(
        cls,
        query: str,
        university_slug: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
    ):
        cls._init_models()
        if cls._chroma_client is False:
            return []

        if query_embedding is None:
            query_embedding = (await asyncio.to_thread(cls._embedding_function, [query]))[0]

        collections_to_query = cls._resolve_collections(university_slug, query_embedding)
        if not collections_to_query:
            return []

        # Pass 1 dense retrieval
        dense_results = await cls._dense_query_all(collections_to_query, query_embedding)

        def _extract_best_distance(results_list) -> float:
            best = None
            for r in results_list:
                dists = (r.get("distances") or [[]])[0]
                for d in dists:
                    if d is None:
                        continue
                    best = d if best is None else min(best, d)
            return best if best is not None else 1e9

        best_distance = _extract_best_distance(dense_results)

        # Optional HyDE second pass only when initial retrieval looks weak
        if cls._should_use_hyde(query) and best_distance > settings.hyde_trigger_distance:
            hyde = await cls._get_or_generate_hyde(query)
            if hyde and hyde.strip():
                hybrid_query = f"{query}\n\n{hyde.strip()}"
                hybrid_embedding = (await asyncio.to_thread(cls._embedding_function, [hybrid_query]))[0]
                dense_results = await cls._dense_query_all(collections_to_query, hybrid_embedding)

        # Flatten dense results into a de-duplicated map of doc -> {meta, best_distance}
        doc_best: Dict[str, Dict[str, Any]] = {}
        for r in dense_results:
            docs = (r.get("documents") or [[]])[0]
            metas = (r.get("metadatas") or [[]])[0]
            dists = (r.get("distances") or [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                if not doc:
                    continue
                if dist is None:
                    dist = 1e9
                prev = doc_best.get(doc)
                if prev is None or dist < prev["distance"]:
                    doc_best[doc] = {"metadata": meta or {}, "distance": float(dist)}

        if not doc_best:
            return []

        dense_ranked = sorted(doc_best.items(), key=lambda kv: kv[1]["distance"])
        dense_ranked_docs = [d for d, _ in dense_ranked]

        # BM25 over dense candidates only (fast, no collection.get)
        bm25_ranked_docs: List[str] = []
        try:
            tokenized_corpus = [doc.lower().split() for doc in dense_ranked_docs]
            bm25 = BM25Okapi(tokenized_corpus)
            q_tokens = query.lower().split()
            scores = bm25.get_scores(q_tokens)
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: settings.bm25_k]
            bm25_ranked_docs = [dense_ranked_docs[i] for i in top_idx if scores[i] > 0]
        except Exception:
            bm25_ranked_docs = []

        # Reciprocal Rank Fusion
        fused_scores: Dict[str, float] = {}
        for r, d in enumerate(dense_ranked_docs):
            fused_scores[d] = fused_scores.get(d, 0.0) + 1 / (60 + r)
        for r, d in enumerate(bm25_ranked_docs):
            fused_scores[d] = fused_scores.get(d, 0.0) + 1 / (60 + r)

        fused_candidates = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        fused_candidates = fused_candidates[: settings.fused_k]
        if not fused_candidates:
            return []

        # Cross-encoder reranking (optional)
        if getattr(cls, "_reranker_failed", False) or not cls._reranker:
            top_pairs = [(doc, float(1.0 - (i * 0.1))) for i, doc in enumerate(fused_candidates[: settings.final_k])]
        else:
            try:
                cross_inp = [[query, doc] for doc in fused_candidates]
                rerank_scores = await asyncio.to_thread(cls._reranker.predict, cross_inp)
                doc_score_pairs = list(zip(fused_candidates, rerank_scores))
                doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
                top_pairs = doc_score_pairs[: settings.final_k]
            except Exception as e:
                print(f"Warning: reranker prediction failed, falling back to fusion rank. Error: {e}")
                top_pairs = [(doc, float(1.0 - (i * 0.1))) for i, doc in enumerate(fused_candidates[: settings.final_k])]

        final_results = []
        for doc_text, score in top_pairs:
            meta = doc_best.get(doc_text, {}).get("metadata", {})
            final_results.append({"text": doc_text, "score": float(score), "metadata": meta})

        return final_results
