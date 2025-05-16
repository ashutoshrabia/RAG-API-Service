from fastapi import APIRouter, HTTPException, UploadFile, File
import shutil, stat, git, os
from pathlib import Path
from pydantic import BaseModel
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import PyPDF2
from .vector_store import save_faiss_index, get_faiss_index, load_documents


router = APIRouter()
BASE_DIR = Path("tmp/gs_docs")

# Windows-safe rmtree error handler
def _on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def get_clip_model():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor

def process_text_file(file_path: str, clip_model, clip_processor):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    inputs = clip_processor(text=content, return_tensors="pt", truncation=True, padding=True)
    embedding = clip_model.get_text_features(**inputs).detach().numpy()
    return content, embedding

def process_image_file(file_path: str, clip_model, clip_processor):
    image = Image.open(file_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    embedding = clip_model.get_image_features(**inputs).detach().numpy()
    return f"Image: {file_path}", embedding

def process_pdf_file(file_path: str, clip_model, clip_processor):
    with open(file_path, "rb") as f:
        pdf = PyPDF2.PdfReader(f)
        content = ""
        for page in pdf.pages:
            content += page.extract_text() or ""
    inputs = clip_processor(text=content, return_tensors="pt", truncation=True, padding=True)
    embedding = clip_model.get_text_features(**inputs).detach().numpy()
    return content, embedding

class IngestGitRequest(BaseModel):
    repo_url: str
    branch:  str = "main"

@router.post("/git")
async def ingest_git(req: IngestGitRequest):
    repo_url, branch = req.repo_url, req.branch

    if not repo_url:
        raise HTTPException(status_code=400, detail="`repo_url` is required")

    clip_model, clip_processor = get_clip_model()

    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_path = BASE_DIR / repo_name

    
    if repo_path.exists():
        shutil.rmtree(repo_path, onerror=_on_rm_error)
    repo_path.mkdir(parents=True, exist_ok=True)


    try:
        git.Repo.clone_from(repo_url, str(repo_path), branch=branch, depth=1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Git clone failed: {e}")

    documents = []
    embeddings = []
    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue
        file_str = str(file_path)
        try:
            if file_str.endswith((".txt", ".md")):
                content, emb = process_text_file(file_str, clip_model, clip_processor)
            elif file_str.lower().endswith((".png", ".jpg", ".jpeg")):
                content, emb = process_image_file(file_str, clip_model, clip_processor)
            elif file_str.endswith(".pdf"):
                content, emb = process_pdf_file(file_str, clip_model, clip_processor)
            else:
                continue

            documents.append({"source": file_str, "content": content})
            embeddings.append(emb)
        except Exception as e:
            print(f"Error processing {file_str}: {e}")
            continue

    if not documents:
        raise HTTPException(status_code=400, detail="No valid documents found to ingest")

   
    matrix = np.vstack(embeddings)
    dim = matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(matrix)

    save_faiss_index(index, documents)

    return {"status": "ingestion complete","documents": len(documents)}

@router.post("/upload")
async def ingest_upload(file: UploadFile = File(...)):
    upload_dir = BASE_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename


    with file_path.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    clip_model, clip_processor = get_clip_model()

    file_str = str(file_path)
    try:
        if file_str.endswith((".txt", ".md")):
            content, emb = process_text_file(file_str, clip_model, clip_processor)
        elif file_str.lower().endswith((".png", ".jpg", ".jpeg")):
            content, emb = process_image_file(file_str, clip_model, clip_processor)
        elif file_str.endswith(".pdf"):
            content, emb = process_pdf_file(file_str, clip_model, clip_processor)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        # emb may be shape (1, D) or (D,), so flatten/squeeze to (D,)
        emb = np.asarray(emb).reshape(-1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

  
    try:
        index     = get_faiss_index()
        documents = load_documents()
        existing  = index.reconstruct_n(0, index.ntotal)
        # combine existing (N, D) with new emb (D,)
        matrix    = np.vstack([existing, emb])
        index     = faiss.IndexFlatL2(matrix.shape[1])
        index.add(matrix)
        documents.append({"source": file_str, "content": content})
    except FileNotFoundError:
        # first upload: create index of dimension D
        index     = faiss.IndexFlatL2(emb.shape[0])
        index.add(np.expand_dims(emb, 0))
        documents = [{"source": file_str, "content": content}]
 
    save_faiss_index(index, documents)
    return {"status": "upload ingestion complete", "document": file_str}
