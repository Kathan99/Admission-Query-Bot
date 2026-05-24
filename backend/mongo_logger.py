"""
mongo_logger.py
---------------
Handles all MongoDB Atlas write operations for query/answer logging.

Design principles:
  - Lazy connection: client is created on first write, not at import time.
    This means a missing/wrong MONGODB_URI won't crash startup.
  - Fire-and-forget: every write is wrapped in try/except so a Mongo
    outage or network blip never breaks a chat response.
  - Feature flag: if ENABLE_MONGO_LOGGING=false (or MONGODB_URI is empty),
    all calls are instant no-ops.
  - Index setup: indexes are created once on the first successful connection
    so the collection is always query-ready.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

try:
    from backend.config import settings
except ModuleNotFoundError:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.config import settings


class MongoLogger:
    _client = None          # pymongo.MongoClient — shared across threads
    _collection = None      # pymongo.Collection
    _lock = threading.Lock()
    _indexes_created = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _get_collection(cls):
        """Return the pymongo Collection, creating the client lazily on first call."""
        if cls._collection is not None:
            return cls._collection

        with cls._lock:
            # Double-checked locking — another thread may have initialised while we waited
            if cls._collection is not None:
                return cls._collection

            if not settings.mongodb_uri:
                print("[MongoLogger] MONGODB_URI is not set — logging disabled.")
                return None

            try:
                from pymongo import MongoClient, ASCENDING, DESCENDING
                from pymongo.server_api import ServerApi

                client = MongoClient(
                    settings.mongodb_uri,
                    server_api=ServerApi("1"),
                    # Short timeouts so a bad URI fails fast, not in the request path
                    connectTimeoutMS=5000,
                    serverSelectionTimeoutMS=5000,
                )
                # Ping to verify connectivity at startup time
                client.admin.command("ping")
                print("[MongoLogger] Connected to MongoDB Atlas ✓")

                db = client[settings.mongodb_db]
                collection = db[settings.mongodb_collection]

                # Create indexes (idempotent — safe to call multiple times)
                if not cls._indexes_created:
                    collection.create_index([("session_id", ASCENDING)])
                    collection.create_index([("timestamp", DESCENDING)])
                    collection.create_index([("mode", ASCENDING)])
                    collection.create_index([("latency_ms", DESCENDING)])
                    cls._indexes_created = True
                    print(f"[MongoLogger] Indexes ready on '{settings.mongodb_db}.{settings.mongodb_collection}'")

                cls._client = client
                cls._collection = collection
                return cls._collection

            except Exception as e:
                print(f"[MongoLogger] Failed to connect to MongoDB Atlas: {e}")
                return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def log(
        cls,
        *,
        session_id: str,
        original_query: str,
        sanitized_query: str,
        contextualized_query: str,
        answer: str,
        mode: str,
        university_slug: str,
        sources: list[dict],
        retrieved_chunk_count: int,
        top_rerank_score: float | None,
        cache_hit: bool,
        tokens_used: int,
        latency_ms: int,
        timings: dict[str, Any],
    ) -> None:
        """
        Write a single chat-turn document to MongoDB Atlas.

        All failures are silently swallowed — this must never raise.
        """
        if not settings.enable_mongo_logging:
            return

        collection = cls._get_collection()
        if collection is None:
            return

        document = {
            "timestamp": datetime.now(timezone.utc),
            "session_id": session_id,
            "original_query": original_query,
            "sanitized_query": sanitized_query,
            "contextualized_query": contextualized_query,
            "answer": answer,
            "mode": mode,
            "university_slug": university_slug,
            "sources": sources,
            "retrieved_chunk_count": retrieved_chunk_count,
            "top_rerank_score": float(top_rerank_score) if top_rerank_score is not None else None,
            "cache_hit": cache_hit,
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
            "timings": timings,
        }

        try:
            collection.insert_one(document)
        except Exception as e:
            print(f"[MongoLogger] insert_one failed: {e}")
