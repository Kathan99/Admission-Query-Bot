---
title: University Admission RAG Chatbot
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# University Admission RAG Chatbot

A production-grade, multi-university RAG (Retrieval-Augmented Generation) chatbot. It supports serving multiple universities, leveraging specific knowledge extracted from university documents (PDFs, TXT, CSV), while maintaining fallback abilities using the general knowledge of large language models for generic questions.

## Project Structure

- `backend/`: FastAPI application containing all Logic (Knowledge routing, Semantic caching, Guardrails, ingestion).
- `frontend/`: Standalone `index.html` chat interface.
- `data/`: Folder to hold university-specific documents.
- `chroma_db/`: Persistent vector store generated after ingestion.

## Setup & Execution

**1. Set your Environment Configuration**
First, declare your `GROQ_API_KEY`:

```bash
mv .env.example .env
# Edit .env and supply your valid Groq API Key
```

**2. Install Requirements**

Ensure you have a modern version of Python (>3.9). Install all necessary requirements:

```bash
pip install -r requirements.txt
```

**3. Ingesting Documents**

To "feed" documents to a university bot:
1. Ensure the university is listed in `universities.json`. A few exist by default (e.g. `iitbombay`, `iitdelhi`, `dtudelhi`).
2. Drop your target PDFs, TXT, or CSVs into `data/{slug}/` (e.g., `data/iitbombay/`).
3. Run the ingestion process to generate embeddings and map the Vector Database. Be aware that the first time you execute this, it will download the embedding model.

```bash
python backend/ingestion.py --university iitbombay
```

**4. Running the Chat Server**

Run the FastAPI Uvicorn engine to start your chatbot:

```bash
uvicorn backend.main:app --reload
```

Then visit your live, responsive, visually stunning Chat App at:
**[http://localhost:8000/frontend/index.html](http://localhost:8000/frontend/index.html)**

## How The Knowledge Router Works

The core of this system is the `KnowledgeRouter`, determining exactly *how* Groq should respond based on your query:

- **RAG_ONLY**: Active when questions require highly specific facts (seats, exact fees, cutoffs), AND document retrieval confidence is extremely high. The AI strictly quotes only what's in the text and prevents factual hallucinations.
- **LLM_ONLY**: Active for broad career mapping or generalized questions (e.g., "scope of B.Tech CSE") where documents typically fail.
- **BLEND**: The Default Mode. The bot weaves retrieved facts (e.g. application deadlines) with its inherent conversational general knowledge.

Additionally, **Guardrails** automatically sanitize incoming requests (checking off-topic bounds, neutralizing PII details) and ensuring specific metrics are actually verifiable from the embedded chunks.
