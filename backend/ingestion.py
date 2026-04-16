import os
import argparse
import uuid
from datetime import datetime
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import pytesseract
from pdf2image import convert_from_path

# To allow relative imports if run as a script or module
try:
    from backend.config import settings, load_universities
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.config import settings, load_universities

def extract_text_from_pdf(filepath):
    text_data = []
    try:
        reader = PdfReader(filepath)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            
            # Fallback to OCR if less than 50 characters were extracted
            if not text or len(text.strip()) < 50:
                print(f"Page {i+1} seems to be an image. Running OCR...")
                try:
                    # Extract specifically this page (1-indexed for pdf2image)
                    images = convert_from_path(filepath, first_page=i+1, last_page=i+1)
                    if images:
                        text = pytesseract.image_to_string(images[0])
                except Exception as e:
                    print(f"OCR failed for page {i+1} of {filepath}: {e}")
                    text = "" # Fallback to empty text for this page
            
            if text and text.strip():
                text_data.append({"text": text, "page": i + 1})
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
    return text_data

def extract_text_from_txt(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [{"text": f.read(), "page": 1}]
    except Exception as e:
        print(f"Error reading TXT {filepath}: {e}")
        return []

def extract_text_from_csv(filepath):
    import csv
    text_data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                text_data.append({"text": ", ".join(row), "page": i + 1})
    except Exception as e:
        print(f"Error reading CSV {filepath}: {e}")
    return text_data

def generate_global_context(filename: str, full_text: str) -> str:
    try:
        client = Groq(api_key=settings.groq_api_key)
        prompt = (
            f"You are an expert academic document analyzer. I am providing you with the first few pages of a document named '{filename}'. "
            "Please provide a strict, 2-sentence summary of this document, and a comma-separated list of the 10 most important "
            "keywords, syllabus topics, or headers found within it. "
            "Format your response exactly like this:\n"
            "Summary: [Your 2 sentence summary]\n"
            "Keywords: [10 comma separated keywords]\n\n"
            f"Document Text:\n{full_text[:8000]}"
        )
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Failed to generate context for {filename}: {e}")
        return f"Summary: Extracted from {filename}\nKeywords: {filename}"

def ingest_university(university_slug):
    universities = load_universities()
    uni_meta = next((u for u in universities if u["slug"] == university_slug), None)
    if not uni_meta:
        print(f"University '{university_slug}' not found in universities.json.")
        return False
    
    uni_data_dir = os.path.join(settings.data_dir, university_slug)
    os.makedirs(uni_data_dir, exist_ok=True)
    
    # Initialize splitter (larger chunks retain table context and syllabus lists better)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    
    # Initialize Chroma Client and Collection
    chroma_client = chromadb.PersistentClient(path=settings.chroma_db_dir)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    collection_name = f"uni_{university_slug}"
    
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    try:
        collection = chroma_client.get_or_create_collection(
            name=collection_name, 
            embedding_function=sentence_transformer_ef,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        print(f"Failed to create collection {collection_name}: {e}")
        return False

    documents = []
    metadatas = []
    ids = []
    
    # Function to add chunk
    def add_chunk(chunk_text, source_meta):
        documents.append(chunk_text)
        metadatas.append(source_meta)
        ids.append(str(uuid.uuid4()))

    # 1. Ingest registry metadata directly
    registry_text = (
        f"University Name: {uni_meta.get('name')}\n"
        f"Location: {uni_meta.get('location')}\n"
        f"Type: {uni_meta.get('type')}\n"
        f"Affiliating Body: {uni_meta.get('affiliating_body')}\n"
        f"Entrance Exams: {', '.join(uni_meta.get('entrance_exams', []))}\n"
        f"Known For: {', '.join(uni_meta.get('known_for', []))}\n"
        f"Admissions Email: {uni_meta.get('admissions_email')}\n"
        f"Admissions Phone: {uni_meta.get('admissions_phone')}\n"
        f"Website: {uni_meta.get('website')}"
    )
    add_chunk(registry_text, {
        "university_slug": university_slug,
        "filename": "universities.json",
        "page_number": 1,
        "chunk_index": 0,
        "ingested_at": datetime.now().isoformat()
    })

    # 2. Iterate over files
    files = [f for f in os.listdir(uni_data_dir) if os.path.isfile(os.path.join(uni_data_dir, f))]
    for file in files:
        filepath = os.path.join(uni_data_dir, file)
        ext = file.split('.')[-1].lower()
        extracted_pages = []
        
        if ext == 'pdf':
            extracted_pages = extract_text_from_pdf(filepath)
        elif ext == 'txt':
            extracted_pages = extract_text_from_txt(filepath)
        elif ext == 'csv':
            extracted_pages = extract_text_from_csv(filepath)
        else:
            continue
            
        full_text = " ".join([p["text"] for p in extracted_pages])
        if not full_text.strip():
            continue
            
        print(f"Generating global context for {file} via LLM...")
        global_context = generate_global_context(file, full_text)
            
        for page_data in extracted_pages:
            chunks = text_splitter.split_text(page_data["text"])
            for idx, chunk in enumerate(chunks):
                enriched_chunk = f"--- Document Context ---\nFile: {file}\n{global_context}\n\n--- Page Content ---\n{chunk}"
                
                add_chunk(enriched_chunk, {
                    "university_slug": university_slug,
                    "filename": file,
                    "page_number": page_data["page"],
                    "chunk_index": idx,
                    "ingested_at": datetime.now().isoformat()
                })
                
    # 3. Store in Chroma
    if documents:
        # Batch insert into Chroma (recommended batch size is < 41666 for sqlite)
        batch_size = 5000
        for i in range(0, len(documents), batch_size):
            collection.upsert(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        print(f"Successfully ingested {len(documents)} chunks for {university_slug}.")
    else:
        print(f"No documents processed for {university_slug}.")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest university documents")
    parser.add_argument("--university", required=True, help="The slug of the university to ingest, or 'all' to ingest all")
    args = parser.parse_args()
    
    if args.university.lower() == 'all':
        universities = load_universities()
        for uni in universities:
            print(f"\n=== Ingesting {uni['name']} ({uni['slug']}) ===")
            ingest_university(uni['slug'])
    else:
        ingest_university(args.university)
