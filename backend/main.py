import os
import re
import json
import time
import asyncio
import hashlib
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import lancedb as _lancedb
import redis

# To allow relative imports if run as a script or module
try:
    from backend.config import settings, load_universities
    from backend.ingestion import ingest_university, ingest_corpus
    from backend.retrieval import RetrievalManager
    from backend.knowledge_router import KnowledgeRouter
    from backend.generation import GenerationManager
    from backend.cache import semantic_cache
    from backend.guardrails import Guardrails
except ModuleNotFoundError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.config import settings, load_universities
    from backend.ingestion import ingest_university, ingest_corpus
    from backend.retrieval import RetrievalManager
    from backend.knowledge_router import KnowledgeRouter
    from backend.generation import GenerationManager
    from backend.cache import semantic_cache
    from backend.guardrails import Guardrails

app = FastAPI(title="Multi-University RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    # Pre-load models at startup so the first request is fast
    print("Pre-loading AI models...")
    try:
        RetrievalManager._init_models()
        print("AI Models loaded successfully.")
    except Exception as e:
        print(f"Error pre-loading models: {e}")


os.makedirs(settings.log_dir, exist_ok=True)

import concurrent.futures

# --- Ingestion status tracker (in-memory, single process) ---
_ingestion_status: dict = {"state": "idle", "filename": None, "message": None}

# Use a process pool for heavy ingestion tasks so PyTorch uses all CPU cores
_process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)

async def _run_ingestion_with_status(fn, *args, filename: str = None, **kwargs):
    """Wrapper: update _ingestion_status around a blocking ingestion call."""
    global _ingestion_status
    _ingestion_status = {"state": "processing", "filename": filename, "message": f"Processing '{filename}'…"}
    try:
        from functools import partial
        func = partial(fn, *args, **kwargs)
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(_process_pool, func)
        if ok:
            _ingestion_status = {"state": "done", "filename": filename, "message": f"'{filename}' ingested successfully."}
        else:
            _ingestion_status = {"state": "error", "filename": filename, "message": f"Ingestion failed for '{filename}'."}
    except Exception as exc:
        _ingestion_status = {"state": "error", "filename": filename, "message": str(exc)}

# Conversation Memory Store using Redis
# Default to localhost for dev; in docker set REDIS_HOST=redis
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)


class ChatRequest(BaseModel):
    query: str
    # Legacy field kept for backward compatibility; ignored in single-corpus mode.
    university_slug: Optional[str] = None
    session_id: str


def log_trace(data: dict):
    trace_file = os.path.join(settings.log_dir, "rag_trace.jsonl")
    with open(trace_file, "a") as f:
        f.write(json.dumps(data) + "\n")


def _should_contextualize(query: str, chat_history: list) -> bool:
    if not settings.enable_contextualize:
        return False
    if not chat_history:
        return False

    q = query.strip().lower()
    if len(q.split()) <= 6:
        return True

    if re.search(r"\b(it|this|that|these|those|there|they|them|its|he|she|him|her|same)\b", q):
        return True

    if q.startswith(("and ", "also ", "what about", "tell me more", "more about", "what if")):
        return True

    return False


def _ctx_cache_key(session_id: str, query: str, chat_history: list) -> str:
    history_tail = chat_history[-6:] if chat_history else []
    payload = json.dumps({"q": query, "h": history_tail}, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"ctx:v1:{session_id}:{h}"


@app.get("/")
def read_root():
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/frontend/index.html")


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/universities")
def get_universities():
    return load_universities()


@app.post("/api/admin/upload")
async def admin_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Admin upload: user just uploads a PDF.

    In single-corpus mode, we store everything directly under data/ and ingest into one collection.
    """
    try:
        os.makedirs(settings.data_dir, exist_ok=True)
        file_bytes = await file.read()
        file_path = os.path.join(settings.data_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        # Mirror to S3 when running in production with S3 storage enabled
        if settings.use_s3:
            from backend.s3_storage import upload_bytes_to_s3
            s3_key = f"{settings.s3_prefix}{file.filename}"
            upload_bytes_to_s3(file_bytes, s3_key)

        # Incremental ingestion: only process this uploaded file, do not reset collection
        if settings.single_corpus:
            background_tasks.add_task(
                _run_ingestion_with_status, ingest_corpus, False, [file.filename], filename=file.filename
            )
        else:
            # legacy behavior
            target_slug = settings.default_upload_slug
            background_tasks.add_task(
                _run_ingestion_with_status, ingest_university, target_slug, False, [file.filename], filename=file.filename
            )

        return {
            "status": "success",
            "message": f"Document '{file.filename}' uploaded and queued for background ingestion.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/ingestion-status")
def ingestion_status():
    return JSONResponse(_ingestion_status)


@app.post("/ingest")
def ingest(university: Optional[str] = None):
    """Trigger ingestion.

    - If SINGLE_CORPUS=true (default), ingests data/ root into one collection.
    - If a university slug is provided (legacy), ingests that subfolder/collection.
    """
    if settings.single_corpus and not university:
        success = ingest_corpus(reset_collection=True)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to ingest corpus")
        return {"status": "success", "collection": settings.corpus_collection}

    if not university:
        raise HTTPException(status_code=400, detail="Missing university")

    success = ingest_university(university, reset_collection=True)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to ingest {university}")
    return {"status": "success", "university": university}


@app.get("/universities/{slug}/status")
def get_university_status(slug: str):
    try:
        db = _lancedb.connect(settings.lancedb_dir)
        tname = f"uni_{slug}"
        if tname in db.table_names():
            table = db.open_table(tname)
            count = table.count_rows()
            return {"university_slug": slug, "chunk_count": count, "doc_count": count, "last_ingested": datetime.now().isoformat()}
        else:
            return {"university_slug": slug, "chunk_count": 0, "doc_count": 0, "last_ingested": None}
    except Exception:
        return {"university_slug": slug, "chunk_count": 0, "doc_count": 0, "last_ingested": None}


def _extract_and_filter_sources(answer_text: str, all_sources: list) -> tuple[str, list]:
    """Strip the LLM-generated References section from answer_text and use the
    filenames it cited to filter all_sources down to only what was actually used.
    Returns (cleaned_answer, filtered_sources).
    """
    # Split on the separator the prompt instructs the LLM to use
    parts = re.split(r"\n---\s*\n\*\*References\*\*", answer_text, maxsplit=1)
    if len(parts) == 1:
        # No References section found — return as-is with no sources shown
        return answer_text.rstrip(), []

    clean_answer = parts[0].rstrip()
    refs_block = parts[1]

    # Extract filenames mentioned in the References block (lines like "1. <filename> — Page N")
    cited_filenames: set[str] = set()
    for line in refs_block.splitlines():
        # Match "1. filename — Page N" or "1. filename"
        m = re.match(r"\s*\d+\.\s+(.+?)(?:\s+[—-]+\s+.*)?$", line.strip())
        if m:
            cited_filenames.add(m.group(1).strip())

    if not cited_filenames:
        return clean_answer, []

    # Keep only sources whose filename appears in the LLM's cited set
    filtered = [s for s in all_sources if s.get("filename", "") in cited_filenames]
    # De-duplicate by filename
    seen: set[str] = set()
    deduped = []
    for s in filtered:
        fn = s.get("filename", "")
        if fn not in seen:
            seen.add(fn)
            deduped.append(s)

    return clean_answer, deduped


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    t_start = time.perf_counter()
    timings = {
        "t_history_ms": 0,
        "t_contextualize_ms": 0,
        "t_embed_ms": 0,
        "t_retrieve_ms": 0,
        "t_generate_ms": 0,
    }

    cache_slug = "corpus" if settings.single_corpus else (request.university_slug if request.university_slug else "global_search")

    # Guardrails: off topic
    if Guardrails.is_off_topic(request.query):
        universities = load_universities()
        uni_name = "these universities"
        if (not settings.single_corpus) and request.university_slug:
            uni_meta = next((u for u in universities if u.get("slug") == request.university_slug), {})
            uni_name = uni_meta.get("name", "this university")

        latency_ms = int((time.perf_counter() - t_start) * 1000)
        return JSONResponse(
            {
                "answer": Guardrails.construct_off_topic_response(uni_name),
                "mode": "GUARDRAIL",
                "sources": [],
                "university": cache_slug,
                "cached": False,
                "latency_ms": latency_ms,
            }
        )

    sanitized_query = Guardrails.strip_pii(request.query)

    # Fetch history
    t0 = time.perf_counter()
    chat_history = []
    try:
        cached_history = redis_client.get(request.session_id)
        if cached_history:
            chat_history = json.loads(cached_history)
    except Exception as e:
        print(f"Redis fetch failed: {e}")
    timings["t_history_ms"] = int((time.perf_counter() - t0) * 1000)

    # Contextualize (heuristic + cached)
    t0 = time.perf_counter()
    contextualized_query = sanitized_query
    if _should_contextualize(sanitized_query, chat_history):
        key = _ctx_cache_key(request.session_id, sanitized_query, chat_history)
        try:
            cached_ctx = redis_client.get(key)
        except Exception:
            cached_ctx = None

        if cached_ctx:
            contextualized_query = cached_ctx
        else:
            contextualized_query = await asyncio.to_thread(
                GenerationManager.contextualize_query, sanitized_query, chat_history
            )
            try:
                redis_client.set(key, contextualized_query, ex=settings.contextualize_cache_ttl_seconds)
            except Exception:
                pass
    timings["t_contextualize_ms"] = int((time.perf_counter() - t0) * 1000)

    # Embed once for cache + retrieval (off the event loop — CPU-bound)
    RetrievalManager._init_models()
    t0 = time.perf_counter()
    query_embedding = (await asyncio.to_thread(RetrievalManager._embedding_function, [contextualized_query]))[0]
    timings["t_embed_ms"] = int((time.perf_counter() - t0) * 1000)

    cached_result = semantic_cache.get(cache_slug, query_embedding)
    if cached_result:
        latency_ms = int((time.perf_counter() - t_start) * 1000)
        log_trace(
            {
                "timestamp": datetime.now().isoformat(),
                "session_id": request.session_id,
                "university_slug": cache_slug,
                "query": sanitized_query,
                "mode": cached_result["mode"],
                "cache_hit": True,
                "tokens_used": 0,
                "latency_ms": latency_ms,
                **timings,
            }
        )
        return JSONResponse(
            {
                "answer": cached_result["response"],
                "mode": cached_result["mode"],
                "sources": cached_result["sources"],
                "university": cache_slug,
                "cached": True,
                "latency_ms": latency_ms,
            }
        )

    # Retrieve chunks (reuse query_embedding)
    t0 = time.perf_counter()
    chunks = await RetrievalManager.retrieve(contextualized_query, None if settings.single_corpus else cache_slug, query_embedding=query_embedding)
    timings["t_retrieve_ms"] = int((time.perf_counter() - t0) * 1000)

    top_rerank_score = chunks[0]["score"] if chunks else 0.0

    mode = KnowledgeRouter.route(contextualized_query, top_rerank_score, has_docs=len(chunks) > 0)

    # Format context
    context_text = ""
    sources = []
    for i, c in enumerate(chunks):
        meta = c.get("metadata", {})
        if mode != KnowledgeRouter.MODE_LLM_ONLY:
            filename = meta.get("filename", "Unknown")
            page_number = meta.get("page_number", "Unknown")
            context_text += (
                f"\n--- Document {i+1}: File: {filename} - Page: {page_number} ---\n{c['text']}\n"
            )

        sources.append(
            {
                "filename": meta.get("filename", "Unknown"),
                "page_number": meta.get("page_number", 0),
            }
        )

    # In single-corpus mode we don't bind to a specific university; use generic contact guidance.
    universities = load_universities()
    if (not settings.single_corpus) and request.university_slug:
        uni_meta = next((u for u in universities if u.get("slug") == request.university_slug), {})
    else:
        uni_meta = {
            "name": "Admissions Assistant",
            "admissions_email": "the official website",
            "admissions_phone": "the official website",
            "website": "the official website",
        }

    # Generate response
    t0 = time.perf_counter()
    completion = GenerationManager.generate_response(
        contextualized_query,
        mode,
        context_text,
        uni_meta,
        stream=False,
        chat_history=chat_history,
    )
    timings["t_generate_ms"] = int((time.perf_counter() - t0) * 1000)

    answer_text = completion.choices[0].message.content

    if mode == KnowledgeRouter.MODE_RAG_ONLY:
        answer_text = Guardrails.verify_numbers_in_rag(answer_text, context_text)

    # Parse which sources the LLM actually cited, then strip the inline References
    # section from the answer to avoid duplicating it in the UI's "View Sources" widget.
    answer_text, sources = _extract_and_filter_sources(answer_text, sources)

    semantic_cache.set(cache_slug, query_embedding, answer_text, mode, sources)

    latency_ms = int((time.perf_counter() - t_start) * 1000)
    tokens_used = completion.usage.total_tokens if completion.usage else 0

    log_trace(
        {
            "timestamp": datetime.now().isoformat(),
            "session_id": request.session_id,
            "university_slug": cache_slug,
            "query": contextualized_query,
            "mode": mode,
            "retrieved_chunk_count": len(chunks),
            "top_rerank_score": float(top_rerank_score) if top_rerank_score else None,
            "cache_hit": False,
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
            **timings,
        }
    )

    # Update session memory (max 6 turns / 3 pairs)
    chat_history.append({"role": "user", "content": sanitized_query})
    chat_history.append({"role": "assistant", "content": answer_text})
    if len(chat_history) > 6:
        chat_history = chat_history[-6:]

    try:
        redis_client.set(request.session_id, json.dumps(chat_history), ex=86400)
    except Exception as e:
        print(f"Redis save failed: {e}")

    return JSONResponse(
        {
            "answer": answer_text,
            "mode": mode,
            "sources": sources if mode != KnowledgeRouter.MODE_LLM_ONLY else [],
            "university": cache_slug,
            "cached": False,
            "latency_ms": latency_ms,
        }
    )

app.mount(
    "/frontend",
    StaticFiles(directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")),
    name="frontend",
)
