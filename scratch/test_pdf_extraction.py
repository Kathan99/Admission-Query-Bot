from pypdf import PdfReader
import os

filepath = "data/career-guidance/DU_B.Tech (1).pdf"
if not os.path.exists(filepath):
    print(f"File not found: {filepath}")
else:
    reader = PdfReader(filepath)
    print(f"Total pages: {len(reader.pages)}")
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        print(f"Page {i+1} text length: {len(text) if text else 0}")
        if text:
            print(f"Snippet: {text[:200]}")
