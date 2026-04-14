import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from backend.ingestion import generate_global_context
from pypdf import PdfReader

reader = PdfReader("data/career-guidance/Brochure_BITSAT-2025 (1).pdf")
pages = [p.extract_text() for p in reader.pages]
full_text = " ".join(pages)
print(generate_global_context("Brochure", full_text))
