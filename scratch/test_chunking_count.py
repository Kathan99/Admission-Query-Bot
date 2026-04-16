import os
import uuid
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def test_chunking():
    filepath = "data/career-guidance/DU_B.Tech (1).pdf"
    reader = PdfReader(filepath)
    text_data = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            text_data.append({"text": text, "page": i + 1})
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    
    all_chunks = []
    for page_data in text_data:
        chunks = text_splitter.split_text(page_data["text"])
        all_chunks.extend(chunks)
        print(f"Page {page_data['page']}: {len(chunks)} chunks")
    
    print(f"Total chunks: {len(all_chunks)}")

if __name__ == "__main__":
    test_chunking()
