import os
from sentence_transformers import CrossEncoder
from chromadb.utils import embedding_functions

# To ensure the models are downloaded and cached during docker build
def download_models():
    print("Downloading embedding model...")
    # This will trigger the download of 'all-MiniLM-L6-v2' via sentence_transformers
    embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    print("Downloading cross-encoder model...")
    # This will trigger the download of 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    
    print("Models downloaded and cached successfully.")

if __name__ == "__main__":
    download_models()
