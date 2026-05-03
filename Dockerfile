# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies
# - build-essential: needed for some C-based Python packages
# - tesseract-ocr + tesseract-ocr-eng: Tesseract OCR engine for scanned-page fallback
# - libglib2.0-0: common runtime dep for image processing libs
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Create a non-root user for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user

# Pre-create model cache directory and set permissions for the 'user'
RUN mkdir -p /app/model_cache && chown -R 1000:1000 /app

USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set model cache directories to ensure they are writable by the 'user'
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache HuggingFace Models locally
COPY download_models.py .
RUN python download_models.py

# Copy the rest of the application codebase
# We do this as root then switch back to ensure permissions are correct if needed
# but since we already switched to USER 1000, we should just COPY and ensure ownership
USER root
COPY . .
RUN chown -R user:user /app
USER user

# Expose Hugging Face's default port
EXPOSE 7860

# Command to boot up the FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
