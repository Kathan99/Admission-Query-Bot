import pypdf
import sys

def search_pdf(filepath):
    try:
        reader = pypdf.PdfReader(filepath)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if "Annexure" in text or "Chemistry" in text:
                print(f"--- Page {i+1} Match ---")
                print(text[:400].replace('\n', ' '))
    except Exception as e:
        print(f"Error: {e}")

search_pdf(sys.argv[1])
