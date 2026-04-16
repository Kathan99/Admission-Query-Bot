import chromadb
from backend.config import settings

client = chromadb.PersistentClient(path=settings.chroma_db_dir)
collection = client.get_collection(name="uni_career-guidance")
results = collection.get(
    where={"filename": "DU_B.Tech (1).pdf"},
    include=['documents']
)

print(f"Found {len(results['documents'])} chunks for DU_B.Tech (1).pdf")
for i, doc in enumerate(results['documents'][:5]):
    print(f"--- Chunk {i+1} ---\n{doc[:500]}...\n")
