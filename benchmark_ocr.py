"""Benchmark script: test OCR speed on a sample of scanned pages using ingestion module."""
import os, time, sys

# Set env before importing settings
os.environ["ENABLE_OCR_FALLBACK"] = "true"
os.environ["OCR_THREADS"] = "4"
os.environ["OCR_SCALE"] = "2.0"

# Use the proper module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.ingestion import _ocr_page_worker

import pypdfium2 as pdfium
from concurrent.futures import ProcessPoolExecutor, as_completed

filepath = "data/Exploring_Career_Cards_Volume1 (1).pdf"
PAGES_TO_TEST = list(range(6, 26))  # 20 scanned pages

if __name__ == "__main__":
    print(f"Benchmarking {len(PAGES_TO_TEST)} scanned pages with 4 parallel workers...")
    start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_ocr_page_worker, filepath, p, 2.0) for p in PAGES_TO_TEST]
        for fut in as_completed(futures):
            txt = fut.result()
            if txt:
                results.append(txt)
    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Processed: {len(PAGES_TO_TEST)} pages")
    print(f"  Got text from: {len(results)} pages")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {elapsed/len(PAGES_TO_TEST):.2f}s per page (effective with parallelism)")
    print(f"  Estimated full PDF (472 pages): {elapsed/len(PAGES_TO_TEST)*472/60:.1f} min")
