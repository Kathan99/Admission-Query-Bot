import os
import json
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Optional: Hugging Face token for Docling/HF downloads (also read directly from env by HF Hub).
    hf_token: str = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_HUB_TOKEN", ""))

    # Providers / models
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    hyde_model: str = os.getenv("HYDE_MODEL", "llama-3.1-8b-instant")

    # Single-corpus mode (all files live directly under data/)
    single_corpus: bool = os.getenv("SINGLE_CORPUS", "true").strip().lower() in {"1", "true", "yes"}
    corpus_collection: str = os.getenv("CORPUS_COLLECTION", "uni_corpus")
    corpus_meta_slug: str = os.getenv("CORPUS_META_SLUG", "corpus")

    # Admin upload defaults (legacy; used only when single_corpus=false)
    default_upload_slug: str = os.getenv("DEFAULT_UPLOAD_SLUG", "career-guidance")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Feature flags
    enable_hyde: bool = True
    enable_contextualize: bool = True
    enable_global_context: bool = False  # ingestion-time LLM summary; keep off for speed

    # Retrieval tuning
    dense_k: int = 30
    bm25_k: int = 10
    fused_k: int = 10
    final_k: int = 5
    global_search_top_n: int = 3

    # HyDE gating/caching
    hyde_trigger_distance: float = 0.35  # cosine distance: lower is better
    hyde_max_tokens: int = 150
    hyde_cache_ttl_seconds: int = 3600

    # Contextualize caching
    contextualize_max_tokens: int = 100
    contextualize_cache_ttl_seconds: int = 86400

    storage_dir: str = os.getenv("STORAGE_DIR", os.path.dirname(os.path.dirname(__file__)))
    chroma_db_dir: str = os.path.join(storage_dir, "chroma_db")
    data_dir: str = os.path.join(storage_dir, "data")
    log_dir: str = os.path.join(storage_dir, "logs")
    universities_json: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "universities.json")

    # Ignore unknown keys in .env (e.g., HF_TOKEN), so Settings doesn't crash on extra variables.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()


def load_universities():
    if os.path.exists(settings.universities_json):
        with open(settings.universities_json, "r") as f:
            return json.load(f)
    return []
