import chromadb
from chromadb.utils import embedding_functions
from backend.config import settings

def test():
    client = chromadb.PersistentClient(path=settings.chroma_db_dir)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
    collection = client.get_collection(name="uni_career-guidance", embedding_function=ef)
    
    # Query dense
    results = collection.query(
        query_texts=["What are the specific topics under Chemistry syllabus for BITSAT-2025?"],
        n_results=10,
        include=["documents", "metadatas"]
    )
    
    print("--- CHUNKS RETURNED ---")
    for doc in results["documents"][0]:
        print("===")
        print(doc[:200].replace('\n', ' '))
        print("===")
        
test()
