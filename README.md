# RAG API Service (Godspeed)

## Overview

A FastAPI-based service for embedding and querying documentation (including code, text, images, PDFs) using a vector database (FAISS) and a Retrieval-Augmented Generation (RAG) agent powered by a multimodal library (CLIP).


##  Features

- **Ingest Git Repository**: Clone any public Git repo (and optionally a branch) via `/ingest/git` and embed `.txt`, `.md`, `.pdf`, `.png`, `.jpg` using CLIP.
- **Upload File**: POST a single file to `/ingest/upload` (text, PDF, image) and upsert into FAISS.
- **Vector Database**: Persist embeddings in FAISS (`tmp/vector_store.index`) with document metadata (`tmp/documents.pkl`).
- **Query API**: `/query/` accepts a JSON body (`question`, `top_k`) and returns a RAG-generated answer plus source list and citations.
- **Multimodal Embeddings**: Use CLIPModel for both text and images.
- **Swagger UI**: Interactive API docs at `http://<host>:<port>/docs`.

---

##  Prerequisites

- Python 3.9+
- Git CLI
- (Optional) Docker & Docker Compose
- A Perplexity API Key (set `PERPLEXITY_API_KEY` in `.env`)

---

##  Installation

1. **Clone this repo**
   ```bash
   git clone https://github.com/your-org/assignment.git
   cd assignment

2. **Create & activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Linux/macOS
   .\.venv\Scripts\activate

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt


4. **Setup environment variables**
   ```bash
   cp .env.example .env
 Edit .env and set your PERPLEXITY_API_KEY

## Running Locally
 From project root
 ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 2000 --reload
```
## Docker
1. **Build**
   ```bash
   docker build -t assignment .
   
2. **Run**
   ```bash
   docker run -d --name assignment -p 2000:2000 \
   -e PERPLEXITY_API_KEY="<your_key>" \
   assignment

3. **Swagger UI**
   http://localhost:2000/docs

## Running Locally
 From project root
 ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 2000 --reload
```
## Ingest Endpoints
1. **Ingest Git Repo**
   ```swift
   POST /ingest/git
   Content-Type: application/json

   {
     "repo_url": "https://github.com/your/repo.git",
     "branch":   "main"      # optional, default is default branch
   }

Response
   ```json
   { "status": "Ingestion complete", "documents": <count> }
```
2. **Upload File**
   ```bash
   POST /ingest/upload
   Content-Type: multipart/form-data

   file: <upload your .txt | .md | .pdf | .png | .jpg>
Response
   ```json
   { "status": "Ingestion complete", "document": "<path>" }
```
## Query Endpoint
 ```bash
POST /query/
Content-Type: application/json

{
  "question": "Your question here",
  "top_k": 3          # optional, default = 3
}
```
Response
```json
   {
  "response":      "<LLM answer>",
  "sources":       ["path/to/doc1", "path/to/doc2", ...],
  "web_citations": [ ... ]
}

```
## Configuration
```bash
# .env
PERPLEXITY_API_KEY=your_perplexity_key
BASE_DIR=tmp/gs_docs
INDEX_PATH=tmp/vector_store.index
DOCUMENTS_PATH=tmp/documents.pkl
```
## Metrics and monitoring
1.Health Check: GET /health → { "status": "ok", "version": "0.1.0" } <br>
2.List Routes: GET /routes → shows all registered endpoints.
