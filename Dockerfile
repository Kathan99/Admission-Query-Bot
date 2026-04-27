# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies
# - tesseract-ocr: Needed for processing image-based PDFs
# - poppler-utils: Needed for pdf2image conversion
# - build-essential: Needed for some C-based python packages (like ChromaDB/HNWSLite)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application codebase
COPY . .

# Expose Uvicorn's default port
EXPOSE 8000

# Command to boot up the FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
