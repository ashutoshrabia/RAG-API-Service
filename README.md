# RAG API Service (Godspeed)

## Overview
This project is a Retrieval-Augmented Generation (RAG) API service built with FastAPI, designed to ingest documents from Git repositories, embed them into a vector database, and query them using natural language. It integrates with the Perplexity API for query augmentation. The project is deployed on Render, with some dependencies mocked to fit within the free tier's resource limits.


## Setup Instructions

### Prerequisites
- Python 3.10
- Git
- A Perplexity API key (sign up at `https://www.perplexity.ai` to obtain one)

### Local Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ashutoshrabia/assignment.git
   cd assignment

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

4. **Set Environment Variables**:
   ```bash
   PERPLEXITY_API_KEY=pplx-your-api-key-here

5. **Run the Application Locally**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 2000

## API endpoints 
Health Check 
Endpoint: GET /health
Description: Check the API's health status.
Example
  ```bash
    curl http://localhost:2000/health
```
Response
  ```json
    {"status": "ok", "version": "0.1.0"}
```
Ingest Git Repository
  ```bash
    curl -X POST "http://localhost:2000/ingest/git" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/godspeed-systems/godspeed"}'
```
Response
  ```json
    {"status": "Ingestion complete", "documents": 1}
```

Upload File
  ```bash
    curl -X POST "http://localhost:2000/ingest/upload" \
  -F "file=@/path/to/your/file.pdf"
```
Response
  ```json
    {"status": "Ingestion complete", "document": "file.pdf"}
```
Query Documents
  ```bash
    curl -X POST "http://localhost:2000/query/" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this documentation about?", "top_k": 1}'
```
Response
  ```json
    {
  "response": "This documentation is about...",
  "sources": ["source1"],
  "web_citations": []
}
```
