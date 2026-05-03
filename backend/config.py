import os
import json
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into os.environ so external libraries like huggingface_hub can see it
load_dotenv()

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
    # OCR fallback: Tesseract is called only for image-only pages.
    # Set ENABLE_OCR_FALLBACK=false to use only native text (fastest, fine for digital PDFs).
    enable_ocr_fallback: bool = os.getenv("ENABLE_OCR_FALLBACK", "true").strip().lower() in {"1", "true", "yes"}

    # Retrieval tuning
    dense_k: int = 15          # was 30; fetch fewer candidates — most were discarded anyway
    bm25_k: int = 10
    fused_k: int = 10
    final_k: int = 5
    global_search_top_n: int = 3

    # HyDE gating/caching
    hyde_trigger_distance: float = 0.28  # was 0.35; only run HyDE on truly weak retrievals
    hyde_max_tokens: int = 150
    hyde_cache_ttl_seconds: int = 3600

    # Contextualize caching
    contextualize_max_tokens: int = 100
    contextualize_cache_ttl_seconds: int = 86400

    # AWS S3 — set USE_S3=true to store/fetch PDFs from S3 in production
    use_s3: bool = os.getenv("USE_S3", "false").strip().lower() in {"1", "true", "yes"}
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    s3_prefix: str = os.getenv("S3_PREFIX", "pdfs/")

    storage_dir: str = os.getenv("STORAGE_DIR", os.path.dirname(os.path.dirname(__file__)))
    chroma_db_dir: str = os.path.join(storage_dir, "chroma_db")
    lancedb_dir: str = os.path.join(storage_dir, "lancedb_db")
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
