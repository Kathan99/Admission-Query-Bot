import os
import asyncio
import hashlib
import json
from typing import Optional, Dict, Any, List, Tuple

from sentence_transformers import CrossEncoder, SentenceTransformer
from rank_bm25 import BM25Okapi
import lancedb

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
    _lancedb_client = None
    _embedding_model = None
    _embedding_function = None
    _reranker = None

    _registry_ready = False
    _redis_client = None

    @classmethod
    def _init_models(cls):
        if cls._lancedb_client is None:
            try:
                os.makedirs(settings.lancedb_dir, exist_ok=True)
                cls._lancedb_client = lancedb.connect(settings.lancedb_dir)
            except Exception as e:
                print(f"Warning: LanceDB unavailable ({e}); retrieval will return no docs.")
                cls._lancedb_client = False  # sentinel: disabled

        if cls._lancedb_client is False:
            return

        if not cls._embedding_model:
            cls._embedding_model = SentenceTransformer(settings.embedding_model)
            cls._embedding_function = lambda texts: cls._embedding_model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).tolist()

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
            db = cls._lancedb_client
            universities = load_universities() or []
            if not universities:
                cls._registry_ready = True
                return

            rows: List[Dict[str, Any]] = []
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
                rows.append({
                    "id": slug,
                    "vector": [],
                    "text": doc,
                    "slug": slug,
                    "name": name,
                    "location": location,
                })

            if rows:
                # Embed registry rows
                texts = [r["text"] for r in rows]
                vecs = cls._embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
                for row, vec in zip(rows, vecs):
                    row["vector"] = vec

                if "uni_registry" in db.table_names():
                    table = db.open_table("uni_registry")
                    (table.merge_insert("id")
                     .when_matched_update_all()
                     .when_not_matched_insert_all()
                     .execute(rows))
                else:
                    db.create_table("uni_registry", data=rows)

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
            db = cls._lancedb_client
            if "uni_registry" not in db.table_names():
                return []
            table = db.open_table("uni_registry")
            results = (
                table.search(query_embedding)
                .metric("cosine")
                .limit(settings.global_search_top_n)
                .to_list()
            )
            slugs = [r.get("slug") for r in results if r.get("slug")]
            out: List[str] = []
            for s in slugs:
                if s not in out:
                    out.append(s)
            return out
        except Exception:
            return []

    @classmethod
    def _resolve_tables(cls, university_slug: Optional[str], query_embedding: List[float]):
        """Return list of (table_name, table) tuples to query."""
        db = cls._lancedb_client

        # Single-corpus mode: always query the one shared table.
        if settings.single_corpus:
            tname = settings.corpus_collection
            try:
                if tname not in db.table_names():
                    db.create_table(tname, data=[])
                return [(tname, db.open_table(tname))]
            except Exception:
                return []

        tables = []
        if university_slug and university_slug != "global_search":
            tname = f"uni_{university_slug}"
            if tname in db.table_names():
                try:
                    tables.append((tname, db.open_table(tname)))
                    return tables
                except Exception:
                    return []
            return []

        # Global search: pick top-N using registry.
        selected = cls._select_global_slugs(query_embedding)
        if selected:
            for slug in selected:
                tname = f"uni_{slug}"
                if tname in db.table_names():
                    try:
                        tables.append((tname, db.open_table(tname)))
                    except Exception:
                        continue
            if tables:
                return tables

        # Fallback: all uni_* tables
        for tname in db.table_names():
            if tname.startswith("uni_") and tname != "uni_registry":
                try:
                    tables.append((tname, db.open_table(tname)))
                except Exception:
                    pass
        return tables

    # Keep _resolve_collections as alias so nothing breaks if called externally
    @classmethod
    def _resolve_collections(cls, university_slug: Optional[str], query_embedding: List[float]):
        return cls._resolve_tables(university_slug, query_embedding)

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
    async def _dense_query_all(cls, tables, query_embedding: List[float]):
        """Query all tables and convert LanceDB results to chromadb-like format."""
        _excluded = {"text", "vector", "id", "_distance"}

        async def query_one(tname_table):
            _tname, table = tname_table
            raw = await asyncio.to_thread(
                lambda: (
                    table.search(query_embedding)
                    .metric("cosine")
                    .limit(settings.dense_k)
                    .to_list()
                )
            )
            docs = [r["text"] for r in raw]
            metas = [{k: v for k, v in r.items() if k not in _excluded} for r in raw]
            dists = [r.get("_distance", 1e9) for r in raw]
            return {
                "documents": [docs],
                "metadatas": [metas],
                "distances": [dists],
            }

        return await asyncio.gather(*[query_one(t) for t in tables])

    @classmethod
    async def retrieve(
        cls,
        query: str,
        university_slug: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
    ):
        cls._init_models()
        if cls._lancedb_client is False:
            return []

        if query_embedding is None:
            query_embedding = (await asyncio.to_thread(cls._embedding_function, [query]))[0]

        tables_to_query = cls._resolve_tables(university_slug, query_embedding)
        if not tables_to_query:
            return []

        # Pass 1 dense retrieval
        dense_results = await cls._dense_query_all(tables_to_query, query_embedding)

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
                dense_results = await cls._dense_query_all(tables_to_query, hybrid_embedding)

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
