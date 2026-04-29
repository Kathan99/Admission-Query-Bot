import os
from sentence_transformers import SentenceTransformer, CrossEncoder

# To ensure the models are downloaded and cached during docker build
def download_models():
    print("Downloading embedding model...")
    SentenceTransformer("all-MiniLM-L6-v2")

    print("Downloading cross-encoder model...")
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

    print("Models downloaded and cached successfully.")

if __name__ == "__main__":
    download_models()
