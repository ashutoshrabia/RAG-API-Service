from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import httpx
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from .vector_store import get_faiss_index, search_index

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def build_rag_prompt(question: str, context: list) -> str:
    context_str = "\n\n".join(
        f"Source: {item[0]['source']}\nContent: {item[0].get('content', '')}"
        for item in context
    )
    return f"""Use the following context to answer the question. If you don't know the answer, say so.

Context:
{context_str}

Question: {question}
Answer:"""

@router.post("/", summary="Query RAG", response_model=dict)
async def query_rag(body: QueryRequest):
    
    question = body.question
    top_k   = body.top_k

    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    if not PERPLEXITY_API_KEY:
        raise HTTPException(status_code=500, detail="Missing Perplexity API key")

    inputs = clip_processor(
        text=question,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    question_embedding = clip_model.get_text_features(**inputs).detach().numpy()

    index   = get_faiss_index()
    context = search_index(index, question_embedding, k=top_k)

    if not context:
        return {
            "response": "No relevant documents found to answer the question.",
            "sources": [],
            "web_citations": []
        }

    prompt = build_rag_prompt(question, context)
    payload = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
            {"role": "user",   "content": prompt}
        ],
        "temperature": 0.4
    }
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type":  "application/json"
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.perplexity.ai/chat/completions",
                json=payload,
                headers=headers,
                timeout=30.0
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Perplexity API request failed: {e}")

    return {
        "response":      data["choices"][0]["message"]["content"],
        "sources":       [item[0]["source"] for item in context],
        "web_citations": data.get("citations", [])
    }
