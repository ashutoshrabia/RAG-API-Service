from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from .ingest import router as ingest_router
from .query import router as query_router
import sys

load_dotenv()

app = FastAPI(
    title="RAG API Service",
    description="API for document ingestion and retrieval-augmented generation",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    ingest_router,
    prefix="/ingest",
    tags=["Document Ingestion"],
)

app.include_router(
    query_router,
    prefix="/query",
    tags=["Query"],
)

@app.get("/")
def root():
    return {"message": "RAG API Service"}

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}

@app.get("/routes")
def list_routes():
    routes = []
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            routes.append({
                "path": route.path,
                "methods": list(route.methods)
            })
    return {"routes": routes}

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run("app.main:app", host="0.0.0.0", port=2000, reload=False)
    except KeyboardInterrupt:
        print("Server stopped by user (Ctrl+C). Shutting down gracefully...")
        sys.exit(0)
