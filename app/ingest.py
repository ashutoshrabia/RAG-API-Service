# app/ingest.py
from fastapi import APIRouter, HTTPException, UploadFile, File
import os
import shutil
import git
from pathlib import Path
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import PyPDF2
from .vector_store import save_faiss_index, get_faiss_index, load_documents

router = APIRouter()

# Initialize CLIP model and processor (loaded lazily to avoid Render startup issues)
def get_clip_model():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor

# Directory to store cloned repos
BASE_DIR = Path("tmp/gs_docs")

def clone_repo(repo_url: str, base_dir: Path) -> Path:
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = base_dir / repo_name
    if repo_path.exists():
        shutil.rmtree(repo_path)
    repo_path.mkdir(parents=True, exist_ok=True)
    git.Repo.clone_from(repo_url, repo_path)
    return repo_path

def process_text_file(file_path: str, clip_model, clip_processor) -> tuple:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    inputs = clip_processor(text=content, return_tensors="pt", truncation=True, padding=True)
    embedding = clip_model.get_text_features(**inputs).detach().numpy()
    return content, embedding

def process_image_file(file_path: str, clip_model, clip_processor) -> tuple:
    image = Image.open(file_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    embedding = clip_model.get_image_features(**inputs).detach().numpy()
    return f"Image: {file_path}", embedding

def process_pdf_file(file_path: str, clip_model, clip_processor) -> tuple:
    with open(file_path, "rb") as f:
        pdf = PyPDF2.PdfReader(f)
        content = ""
        for page in pdf.pages:
            content += page.extract_text()
    inputs = clip_processor(text=content, return_tensors="pt", truncation=True, padding=True)
    embedding = clip_model.get_text_features(**inputs).detach().numpy()
    return content, embedding

@router.post("/git")
async def ingest_git(request: dict):
    repo_url = request.get("repo_url")
    if not repo_url:
        raise HTTPException(status_code=400, detail="Repository URL is required")

    # Load CLIP model
    clip_model, clip_processor = get_clip_model()

    # Clone the repository
    repo_path = clone_repo(repo_url, BASE_DIR)

    # Process all files in the repository
    documents = []
    embeddings = []
    for file_path in repo_path.rglob("*"):
        if file_path.is_file():
            file_str = str(file_path)
            try:
                if file_str.endswith((".txt", ".md")):
                    content, embedding = process_text_file(file_str, clip_model, clip_processor)
                elif file_str.endswith((".png", ".jpg", ".jpeg")):
                    content, embedding = process_image_file(file_str, clip_model, clip_processor)
                elif file_str.endswith(".pdf"):
                    content, embedding = process_pdf_file(file_str, clip_model, clip_processor)
                else:
                    continue
                documents.append({"source": file_str, "content": content})
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing {file_str}: {e}")
                continue

    if not documents:
        raise HTTPException(status_code=400, detail="No valid documents found in the repository")

    # Convert embeddings to a numpy array
    embeddings = np.vstack(embeddings)

    # Create or update FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save the index and documents
    save_faiss_index(index, documents)

    return {"status": "Ingestion complete", "documents": len(documents)}

@router.post("/upload")
async def ingest_upload(file: UploadFile = File(...)):
    # Save the uploaded file
    upload_dir = BASE_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load CLIP model
    clip_model, clip_processor = get_clip_model()

    # Process the file based on its type
    file_str = str(file_path)
    try:
        if file_str.endswith((".txt", ".md")):
            content, embedding = process_text_file(file_str, clip_model, clip_processor)
        elif file_str.endswith((".png", ".jpg", ".jpeg")):
            content, embedding = process_image_file(file_str, clip_model, clip_processor)
        elif file_str.endswith(".pdf"):
            content, embedding = process_pdf_file(file_str, clip_model, clip_processor)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

               # Determine embedding dimension
        dimension = embedding.shape[1]

        # Load existing index and documents, or start from scratch
        try:
            index = get_faiss_index()
            documents = load_documents()
        except FileNotFoundError:
            index = faiss.IndexFlatL2(dimension)
            documents = []

        # Add the new document and embedding
        documents.append({"source": file_str, "content": content})
        embeddings = np.vstack([embedding])

        if index.ntotal > 0:
            existing_embeddings = index.reconstruct_n(0, index.ntotal)
            embeddings = np.vstack([existing_embeddings, embeddings])

        # (Re-)build the FAISS index with all embeddings
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Save the updated index and documents
        save_faiss_index(index, documents)

        return {"status": "Ingestion complete", "document": file_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")