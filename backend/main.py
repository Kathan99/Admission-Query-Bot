import os
import json
import time
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb

# To allow relative imports if run as a script or module
try:
    from backend.config import settings, load_universities
    from backend.ingestion import ingest_university
    from backend.retrieval import RetrievalManager
    from backend.knowledge_router import KnowledgeRouter
    from backend.generation import GenerationManager
    from backend.cache import semantic_cache, cosine_similarity
    from backend.guardrails import Guardrails
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.config import settings, load_universities
    from backend.ingestion import ingest_university
    from backend.retrieval import RetrievalManager
    from backend.knowledge_router import KnowledgeRouter
    from backend.generation import GenerationManager
    from backend.cache import semantic_cache, cosine_similarity
    from backend.guardrails import Guardrails

app = FastAPI(title="Multi-University RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import redis

os.makedirs(settings.log_dir, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=settings.chroma_db_dir)

# Conversation Memory Store using Redis
<<<<<<< HEAD
redis_host = os.getenv("REDIS_HOST", "redis")
=======
redis_host = os.getenv("REDIS_HOST", "localhost")
>>>>>>> 8a11d43 (Initialize Global RAG Chatbot Engine and client ingestion pipeline)
redis_port = int(os.getenv("REDIS_PORT", "6379"))
redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)

# Data Models
class ChatRequest(BaseModel):
    query: str
    university_slug: Optional[str] = None
    session_id: str

def log_trace(data: dict):
    trace_file = os.path.join(settings.log_dir, "rag_trace.jsonl")
    with open(trace_file, "a") as f:
        f.write(json.dumps(data) + "\n")

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
    university: str = Form(...)
):
    try:
        # Create directory if it doesn't exist
        university_dir = os.path.join(settings.data_dir, university)
        os.makedirs(university_dir, exist_ok=True)
        
        # Save file to disk
        file_path = os.path.join(university_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
            
        # Dispatch background processing task
        background_tasks.add_task(ingest_university, university)
        
        return {"status": "success", "message": f"Document '{file.filename}' uploaded and queued for background ingestion."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
def ingest(university: str):
    success = ingest_university(university)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to ingest {university}")
    return {"status": "success", "university": university}

@app.get("/universities/{slug}/status")
def get_university_status(slug: str):
    collection_name = f"uni_{slug}"
    try:
        collection = chroma_client.get_collection(name=collection_name)
        count = collection.count()
        return {
            "university_slug": slug,
            "chunk_count": count,
            "doc_count": count, # Approximation
            "last_ingested": datetime.now().isoformat() # We don't track this perfectly
        }
    except Exception:
        return {"university_slug": slug, "chunk_count": 0, "doc_count": 0, "last_ingested": None}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    
    # We use a fallback slug for caching globally
    cache_slug = request.university_slug if request.university_slug else "global_search"
    
    # Check Guardrails - off topic
    if Guardrails.is_off_topic(request.query):
        universities = load_universities()
        uni_name = "these universities"
        if request.university_slug:
            uni_meta = next((u for u in universities if u["slug"] == request.university_slug), {})
            uni_name = uni_meta.get("name", "this university")
            
        return JSONResponse({
            "answer": Guardrails.construct_off_topic_response(uni_name),
            "mode": "GUARDRAIL",
            "sources": [],
            "university": cache_slug,
            "cached": False,
            "latency_ms": int((time.time() - start_time) * 1000)
        })

    # Sanitize PII
    sanitized_query = Guardrails.strip_pii(request.query)

    # Fetch History
    chat_history = []
    try:
        cached_history = redis_client.get(request.session_id)
        if cached_history:
            chat_history = json.loads(cached_history)
    except Exception as e:
        print(f"Redis fetch failed: {e}")
    
    # Contextualize Query
    contextualized_query = GenerationManager.contextualize_query(sanitized_query, chat_history)

    # We need the query embedding to check cache
    RetrievalManager._init_models()
    query_embedding = RetrievalManager._embedding_function([contextualized_query])[0]
    
    cached_result = semantic_cache.get(cache_slug, query_embedding)
    if cached_result:
        latency_ms = int((time.time() - start_time) * 1000)
        log_trace({
            "timestamp": datetime.now().isoformat(),
            "session_id": request.session_id,
            "university_slug": cache_slug,
            "query": sanitized_query,
            "mode": cached_result['mode'],
            "hyde_query": None,
            "retrieved_chunk_count": 0,
            "top_rerank_score": None,
            "cache_hit": True,
            "tokens_used": 0, 
            "latency_ms": latency_ms
        })
        return JSONResponse({
            "answer": cached_result['response'],
            "mode": cached_result['mode'],
            "sources": cached_result['sources'],
            "university": cache_slug,
            "cached": True,
            "latency_ms": latency_ms
        })

    # Retrieve chunks
    chunks = await RetrievalManager.retrieve(contextualized_query, cache_slug)
    top_rerank_score = chunks[0]['score'] if chunks else 0.0
    
    # Decide Mode
    mode = KnowledgeRouter.route(contextualized_query, top_rerank_score, has_docs=len(chunks) > 0)
    
    # Formatting Context
    context_text = ""
    sources = []
    
    for i, c in enumerate(chunks):
        meta = c.get('metadata', {})
        if mode != KnowledgeRouter.MODE_LLM_ONLY:
            filename = meta.get('filename', 'Unknown')
            page_number = meta.get('page_number', 'Unknown')
            uni_tag = meta.get('university_slug', 'Unknown')
            context_text += f"\n--- Document {i+1}: Source: {uni_tag} - File: {filename} - Page: {page_number} ---\n{c['text']}\n"
        
        sources.append({
            "filename": meta.get('filename', 'Unknown'),
            "page_number": meta.get('page_number', 0),
            "university": meta.get('university_slug', 'Unknown')
        })

    # Generate response
    universities = load_universities()
    if request.university_slug:
        uni_meta = next((u for u in universities if u["slug"] == request.university_slug), {})
    else:
        uni_meta = {
            "name": "Global Admissions Assistant",
            "admissions_email": "their official website",
            "admissions_phone": "their official website",
            "website": "their official website"
        }
    
    # We use streaming = False here to build the full response for Guardrails
    # If the requirement meant "Stream the output via server-sent events to frontend", we can yield JSON chunks
    # But for a simpler architecture, we build the string then stream or return block
    # Given the strict response format `{ answer: str... }` I will return JSONResponse
    completion = GenerationManager.generate_response(
        contextualized_query, 
        mode, 
        context_text, 
        uni_meta,
        stream=False,
        chat_history=chat_history
    )
    
    answer_text = completion.choices[0].message.content
    
    # Guardrails checks
    if mode == KnowledgeRouter.MODE_RAG_ONLY:
        answer_text = Guardrails.verify_numbers_in_rag(answer_text, context_text)

    # Set Cache
    semantic_cache.set(
        cache_slug, 
        query_embedding, 
        answer_text, 
        mode, 
        sources
    )

    latency_ms = int((time.time() - start_time) * 1000)
    tokens_used = completion.usage.total_tokens if completion.usage else 0

    log_trace({
        "timestamp": datetime.now().isoformat(),
        "session_id": request.session_id,
        "university_slug": cache_slug,
        "query": contextualized_query,
        "mode": mode,
        "hyde_query": None, # Could store if we extracted it in generation.py
        "retrieved_chunk_count": len(chunks),
        "top_rerank_score": float(top_rerank_score) if top_rerank_score else None,
        "cache_hit": False,
        "tokens_used": tokens_used,
        "latency_ms": latency_ms
    })

    # Update global session memory (store max 6 turns / 3 pairs)
    chat_history.append({"role": "user", "content": sanitized_query})
    chat_history.append({"role": "assistant", "content": answer_text})
    if len(chat_history) > 6:
        chat_history = chat_history[-6:]
        
    try:
        redis_client.set(request.session_id, json.dumps(chat_history), ex=86400)
    except Exception as e:
        print(f"Redis save failed: {e}")

    return JSONResponse({
        "answer": answer_text,
        "mode": mode,
        "sources": sources if mode != KnowledgeRouter.MODE_LLM_ONLY else [],
        "university": cache_slug,
        "cached": False,
        "latency_ms": latency_ms
    })

# Serve static frontend files
<<<<<<< HEAD
app.mount("/data", StaticFiles(directory=settings.data_dir), name="data")
from fastapi.responses import HTMLResponse

@app.get("/admin", response_class=HTMLResponse)
def admin_panel():
    return open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "admin.html")).read()

=======
>>>>>>> 8a11d43 (Initialize Global RAG Chatbot Engine and client ingestion pipeline)
app.mount("/frontend", StaticFiles(directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")), name="frontend")
