import chromadb
from backend.config import settings

client = chromadb.PersistentClient(path=settings.chroma_db_dir)
collections = client.list_collections()
for col in collections:
    print(f"Collection: {col.name}, Count: {col.count()}")
