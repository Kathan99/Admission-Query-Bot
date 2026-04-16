import chromadb
from backend.config import settings

client = chromadb.PersistentClient(path=settings.chroma_db_dir)
collection = client.get_collection(name="uni_career-guidance")
results = collection.get(include=['metadatas'])
filenames = set(m.get('filename') for m in results['metadatas'] if m)
print("Files in uni_career-guidance:")
for f in sorted(filenames):
    print(f"- {f}")
