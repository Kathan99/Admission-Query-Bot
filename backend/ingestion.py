import os
import argparse
import hashlib
import shutil
from datetime import datetime
from typing import Optional, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
import lancedb
from sentence_transformers import SentenceTransformer
from groq import Groq

# PDF extraction uses a fast two-tier approach:
#   1. pypdfium2  — near-instant native text extraction (already in venv as a transitive dep)
#   2. pytesseract — Tesseract 5 OCR, called ONLY for pages where native text is empty/minimal
# This avoids the 4–12 hour Docling+EasyOCR cost on CPU-only AWS instances.

# To allow relative imports if run as a script or module
try:
    from backend.config import settings, load_universities
except ModuleNotFoundError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.config import settings, load_universities


def _ocr_page_worker(filepath: str, page_no: int, scale: float = 2.0) -> str:
    """Module-level OCR worker — must be at top-level so ProcessPoolExecutor can pickle it.

    Opens its own PdfDocument handle (not thread/process-safe to share one document
    across workers) and runs Tesseract on the rendered page image.

    Returns: extracted text string (empty string on failure).
    """
    try:
        import pypdfium2 as pdfium
        import pytesseract

        doc = pdfium.PdfDocument(filepath)
        page = doc[page_no - 1]           # PdfDocument is 0-indexed
        bitmap = page.render(scale=scale)
        pil_img = bitmap.to_pil()
        doc.close()
        return pytesseract.image_to_string(pil_img, lang="eng").strip()
    except Exception as e:
        print(f"  [OCR worker] page {page_no} failed: {e}")
        return ""


def extract_text_from_pdf(filepath: str):
    """Fast two-tier PDF extractor with parallel OCR.

    Tier 1 (pypdfium2) — native text, runs in milliseconds per page.
    Tier 2 (pytesseract) — Tesseract 5 OCR, only triggered when a page yields
    fewer than OCR_THRESHOLD characters (i.e. it's a scanned/image page).
    Scanned pages are processed in parallel to maximise CPU utilisation.

    Returns: [{"text": str, "page": int}, ...]
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Below this char count a page is treated as image-only / scanned.
    OCR_THRESHOLD = 50

    try:
        import pypdfium2 as pdfium
    except ImportError:
        print("pypdfium2 not available — install it with: pip install pypdfium2")
        return []

    try:
        doc = pdfium.PdfDocument(filepath)
    except Exception as e:
        print(f"pypdfium2 failed to open {filepath}: {e}")
        return []

    # --- Pass 1: quick native-text extraction (single-threaded, near-instant) ---
    native_results: dict[int, str] = {}   # page_no -> text
    scanned_pages: list[int] = []         # page indices that need OCR

    for page_no, page in enumerate(doc, start=1):
        try:
            text = page.get_textpage().get_text_range().strip()
        except Exception:
            text = ""
        if len(text) >= OCR_THRESHOLD:
            native_results[page_no] = text
        else:
            scanned_pages.append(page_no)  # 1-indexed

    doc.close()   # Close before parallelism to avoid shared-state issues with pypdfium2

    results: list[dict] = list(
        {"text": t, "page": p} for p, t in native_results.items()
    )

    # --- Pass 2: parallel OCR for scanned pages ---
    if scanned_pages and settings.enable_ocr_fallback:
        scale = settings.ocr_scale
        n_workers = settings.ocr_threads

        print(f"  Running parallel OCR on {len(scanned_pages)} scanned page(s) "
              f"using {n_workers} worker(s)…")

        # ProcessPoolExecutor gives true CPU parallelism (bypasses GIL).
        # _ocr_page_worker is module-level so it can be pickled by multiprocessing.
        import multiprocessing as _mp
        _ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=_ctx) as pool:
            futures = {
                pool.submit(_ocr_page_worker, filepath, pno, scale): pno
                for pno in scanned_pages
            }
            for fut in as_completed(futures):
                ocr_text = fut.result()
                if ocr_text:
                    results.append({"text": ocr_text, "page": futures[fut]})

        print(f"  OCR complete. {len(scanned_pages)} page(s) processed.")

    return sorted(results, key=lambda r: r["page"])


def extract_text_from_txt(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return [{"text": f.read(), "page": 1}]
    except Exception as e:
        print(f"Error reading TXT {filepath}: {e}")
        return []


def extract_text_from_csv(filepath):
    import csv

    text_data = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                text_data.append({"text": ", ".join(row), "page": i + 1})
    except Exception as e:
        print(f"Error reading CSV {filepath}: {e}")
    return text_data


def generate_global_context(filename: str, full_text: str) -> str:
    if not settings.enable_global_context:
        return ""
    if not settings.groq_api_key:
        return ""

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
            model=settings.hyde_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Failed to generate context for {filename}: {e}")
        return ""


def _stable_chunk_id(university_slug: str, filename: str, page: int, chunk_index: int, file_digest: str) -> str:
    raw = f"{university_slug}|{filename}|{file_digest}|{page}|{chunk_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ingest_corpus(reset_collection: bool = True, only_files: Optional[List[str]] = None) -> bool:
    """Ingest documents directly from data/ into a single LanceDB table."""

    data_dir = settings.data_dir
    os.makedirs(data_dir, exist_ok=True)

    # In production, pull any new/updated PDFs from S3 before processing
    if settings.use_s3:
        from backend.s3_storage import sync_s3_to_local
        sync_s3_to_local(data_dir, prefix=settings.s3_prefix)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)

    # Initialize LanceDB and embedding model
    try:
        if not settings.lancedb_dir.startswith("s3://"):
            os.makedirs(settings.lancedb_dir, exist_ok=True)
        db = lancedb.connect(settings.lancedb_dir)
    except Exception as e:
        print(f"Failed to open LanceDB: {e}")
        return False

    model = SentenceTransformer(settings.embedding_model)
    collection_name = settings.corpus_collection

    # Handle reset
    if reset_collection and collection_name in db.table_names():
        db.drop_table(collection_name)

    rows = []

    file_paths: list[str] = []
    for root, dirs, files in os.walk(data_dir):
        # skip hidden folders
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.startswith("."):
                continue
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                file_paths.append(fp)

    if only_files:
        only_set = set(only_files)
        file_paths = [
            fp
            for fp in file_paths
            if os.path.relpath(fp, data_dir) in only_set or os.path.basename(fp) in only_set
        ]

    corpus_slug = settings.corpus_meta_slug

    eligible = [fp for fp in file_paths if fp.split(".")[-1].lower() in {"pdf", "txt", "csv"}]
    print(f"Found {len(eligible)} file(s) to process.")

    for filepath in file_paths:
        file = os.path.relpath(filepath, data_dir)
        ext = file.split(".")[-1].lower()

        if ext not in {"pdf", "txt", "csv"}:
            continue

        print(f"Extracting text from: {file} ...", flush=True)
        extracted_pages = []
        if ext == "pdf":
            extracted_pages = extract_text_from_pdf(filepath)
        elif ext == "txt":
            extracted_pages = extract_text_from_txt(filepath)
        elif ext == "csv":
            extracted_pages = extract_text_from_csv(filepath)

        full_text = " ".join([p["text"] for p in extracted_pages])
        if not full_text.strip():
            print(f"  Skipping {file} (no text extracted).")
            continue

        print(f"  Extracted {len(extracted_pages)} page(s) from {file}.")

        try:
            file_digest = _file_sha1(filepath)
        except Exception:
            file_digest = str(os.path.getmtime(filepath))

        global_context = ""
        if settings.enable_global_context:
            print(f"Generating global context for {file} via LLM...")
            global_context = generate_global_context(file, full_text)

        file_chunk_count = 0
        for page_data in extracted_pages:
            chunks = text_splitter.split_text(page_data["text"])
            for idx, chunk in enumerate(chunks):
                if global_context:
                    enriched_chunk = (
                        f"--- Document Context ---\nFile: {file}\n{global_context}\n\n"
                        f"--- Page Content ---\n{chunk}"
                    )
                else:
                    enriched_chunk = f"File: {file}\nPage: {page_data['page']}\n\n{chunk}"

                rows.append({
                    "id": _stable_chunk_id(corpus_slug, file, page_data["page"], idx, file_digest),
                    "vector": [],  # filled in below
                    "text": enriched_chunk,
                    "university_slug": corpus_slug,
                    "filename": file,
                    "page_number": page_data["page"],
                    "chunk_index": idx,
                    "file_digest": file_digest,
                    "ingested_at": datetime.now().isoformat(),
                })
                file_chunk_count += 1
        print(f"  Chunked into {file_chunk_count} chunk(s). Total so far: {len(rows)}.")

    if rows:
        # Embed in batches
        BATCH_SIZE = 256
        all_texts = [row["text"] for row in rows]
        all_vectors = []
        total_batches = (len(all_texts) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Embedding {len(all_texts)} chunks in {total_batches} batch(es)...")
        for i in range(0, len(all_texts), BATCH_SIZE):
            batch_num = i // BATCH_SIZE + 1
            print(f"  Encoding batch {batch_num}/{total_batches}...", flush=True)
            batch_vecs = model.encode(
                all_texts[i:i + BATCH_SIZE],
                normalize_embeddings=True,
                show_progress_bar=True,
            ).tolist()
            all_vectors.extend(batch_vecs)
        print("Embedding complete.")
        for row, vec in zip(rows, all_vectors):
            row["vector"] = vec

        if collection_name in db.table_names():
            # Incremental upsert
            print(f"Upserting {len(rows)} rows into existing table '{collection_name}'...", flush=True)
            table = db.open_table(collection_name)
            (table.merge_insert("id")
             .when_matched_update_all()
             .when_not_matched_insert_all()
             .execute(rows))
        else:
            print(f"Creating table '{collection_name}' with {len(rows)} rows...", flush=True)
            table = db.create_table(collection_name, data=rows)

        # Build IVF-PQ ANN index for fast vector search at query time.
        # Without this, LanceDB falls back to brute-force scan on every query.
        try:
            print(f"Building ANN index for '{collection_name}' (this may take a while)...", flush=True)
            table = db.open_table(collection_name)
            table.create_index(metric="cosine", replace=True)
            print(f"ANN index built for {collection_name}.")
        except Exception as e:
            print(f"Warning: could not build ANN index ({e}); queries will use brute-force scan.")

        print(f"Successfully ingested/upserted {len(rows)} chunks into {collection_name}.")
    else:
        print(f"No documents processed for collection {collection_name}.")

    return True


def ingest_university(university_slug: str, reset_collection: bool = True, only_files: Optional[List[str]] = None) -> bool:
    """Legacy multi-university ingestion.

    If SINGLE_CORPUS=true, we ingest from data/ root into one collection.
    """

    if settings.single_corpus:
        return ingest_corpus(reset_collection=reset_collection, only_files=only_files)

    universities = load_universities()
    uni_meta = next((u for u in universities if u.get("slug") == university_slug), None)
    if not uni_meta:
        print(f"University '{university_slug}' not found in universities.json.")
        return False

    uni_data_dir = os.path.join(settings.data_dir, university_slug)
    os.makedirs(uni_data_dir, exist_ok=True)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)

    # Initialize LanceDB and embedding model
    try:
        if not settings.lancedb_dir.startswith("s3://"):
            os.makedirs(settings.lancedb_dir, exist_ok=True)
        db = lancedb.connect(settings.lancedb_dir)
    except Exception as e:
        print(f"Failed to open LanceDB: {e}")
        return False

    model = SentenceTransformer(settings.embedding_model)
    collection_name = f"uni_{university_slug}"

    if reset_collection and collection_name in db.table_names():
        db.drop_table(collection_name)

    rows = []

    # Ensure registry meta chunk exists (idempotent)
    registry_text = (
        f"University Name: {uni_meta.get('name')}\n"
        f"Location: {uni_meta.get('location')}\n"
        f"Type: {uni_meta.get('type')}\n"
        f"Affiliating Body: {uni_meta.get('affiliating_body')}\n"
        f"Entrance Exams: {', '.join(uni_meta.get('entrance_exams', []) or [])}\n"
        f"Known For: {', '.join(uni_meta.get('known_for', []) or [])}\n"
        f"Admissions Email: {uni_meta.get('admissions_email')}\n"
        f"Admissions Phone: {uni_meta.get('admissions_phone')}\n"
        f"Website: {uni_meta.get('website')}"
    )
    rows.append({
        "id": _stable_chunk_id(university_slug, "universities.json", 1, 0, "registry"),
        "vector": [],  # filled in below
        "text": registry_text,
        "university_slug": university_slug,
        "filename": "universities.json",
        "page_number": 1,
        "chunk_index": 0,
        "file_digest": "registry",
        "ingested_at": datetime.now().isoformat(),
    })

    files = [f for f in os.listdir(uni_data_dir) if os.path.isfile(os.path.join(uni_data_dir, f))]
    if only_files:
        only_set = set(only_files)
        files = [f for f in files if f in only_set]

    for file in files:
        filepath = os.path.join(uni_data_dir, file)
        ext = file.split(".")[-1].lower()

        if ext not in {"pdf", "txt", "csv"}:
            continue

        extracted_pages = []
        if ext == "pdf":
            extracted_pages = extract_text_from_pdf(filepath)
        elif ext == "txt":
            extracted_pages = extract_text_from_txt(filepath)
        elif ext == "csv":
            extracted_pages = extract_text_from_csv(filepath)

        full_text = " ".join([p["text"] for p in extracted_pages])
        if not full_text.strip():
            continue

        try:
            file_digest = _file_sha1(filepath)
        except Exception:
            file_digest = str(os.path.getmtime(filepath))

        global_context = ""
        if settings.enable_global_context:
            print(f"Generating global context for {file} via LLM...")
            global_context = generate_global_context(file, full_text)

        for page_data in extracted_pages:
            chunks = text_splitter.split_text(page_data["text"])
            for idx, chunk in enumerate(chunks):
                if global_context:
                    enriched_chunk = (
                        f"--- Document Context ---\nFile: {file}\n{global_context}\n\n"
                        f"--- Page Content ---\n{chunk}"
                    )
                else:
                    enriched_chunk = f"File: {file}\nPage: {page_data['page']}\n\n{chunk}"

                rows.append({
                    "id": _stable_chunk_id(university_slug, file, page_data["page"], idx, file_digest),
                    "vector": [],  # filled in below
                    "text": enriched_chunk,
                    "university_slug": university_slug,
                    "filename": file,
                    "page_number": page_data["page"],
                    "chunk_index": idx,
                    "file_digest": file_digest,
                    "ingested_at": datetime.now().isoformat(),
                })

    if rows:
        # Embed in batches
        BATCH_SIZE = 256
        all_texts = [row["text"] for row in rows]
        all_vectors = []
        total_batches = (len(all_texts) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Embedding {len(all_texts)} chunks in {total_batches} batch(es)...")
        for i in range(0, len(all_texts), BATCH_SIZE):
            batch_num = i // BATCH_SIZE + 1
            print(f"  Encoding batch {batch_num}/{total_batches}...", flush=True)
            batch_vecs = model.encode(
                all_texts[i:i + BATCH_SIZE],
                normalize_embeddings=True,
                show_progress_bar=True,
            ).tolist()
            all_vectors.extend(batch_vecs)
        print("Embedding complete.")
        for row, vec in zip(rows, all_vectors):
            row["vector"] = vec

        if collection_name in db.table_names():
            # Incremental upsert
            print(f"Upserting {len(rows)} rows into existing table '{collection_name}'...", flush=True)
            table = db.open_table(collection_name)
            (table.merge_insert("id")
             .when_matched_update_all()
             .when_not_matched_insert_all()
             .execute(rows))
        else:
            print(f"Creating table '{collection_name}' with {len(rows)} rows...", flush=True)
            table = db.create_table(collection_name, data=rows)

        # Build IVF-PQ ANN index for fast vector search at query time.
        try:
            print(f"Building ANN index for '{collection_name}' (this may take a while)...", flush=True)
            table = db.open_table(collection_name)
            table.create_index(metric="cosine", replace=True)
            print(f"ANN index built for {collection_name}.")
        except Exception as e:
            print(f"Warning: could not build ANN index ({e}); queries will use brute-force scan.")

        print(f"Successfully ingested/upserted {len(rows)} chunks for {university_slug}.")
    else:
        print(f"No documents processed for {university_slug}.")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents from the data/ folder into Chroma."
    )
    parser.add_argument(
        "--university",
        required=False,
        default=None,
        help=(
            "Legacy: Optional slug to ingest (e.g., iitbombay). "
            "If omitted, ingests from data/ root into the shared corpus collection."
        ),
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Do not reset collections (upsert only).",
    )
    args = parser.parse_args()

    reset = not args.incremental

    if args.university:
        ingest_university(args.university, reset_collection=reset)
    else:
        # Default: ingest from data/ root into a single shared collection.
        ingest_corpus(reset_collection=reset)
