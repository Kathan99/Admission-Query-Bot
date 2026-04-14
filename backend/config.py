import os
import json
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    storage_dir: str = os.getenv("STORAGE_DIR", os.path.dirname(os.path.dirname(__file__)))
    chroma_db_dir: str = os.path.join(storage_dir, "chroma_db")
    data_dir: str = os.path.join(storage_dir, "data")
    log_dir: str = os.path.join(storage_dir, "logs")
    universities_json: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "universities.json")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()

def load_universities():
    if os.path.exists(settings.universities_json):
        with open(settings.universities_json, "r") as f:
            return json.load(f)
    return []
