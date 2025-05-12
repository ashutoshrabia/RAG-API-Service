from fastapi import APIRouter, HTTPException, Request
import httpx
import os
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from .vector_store import get_faiss_index, search_index

router = APIRouter()

# Initialize CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def build_rag_prompt(question: str, context: list) -> str:
    context_str = "\n\n".join([f"Source: {item[0]['source']}\nContent: {item[0].get('content', '')}" 
                              for item in context])
    return f"""Use the following context to answer the question. If you don't know the answer, say so.
    
Context:
{context_str}

Question: {question}
Answer:"""

@router.post("/")
async def query_rag(request: Request):
    body = await request.json()
    question = body.get("question", "")
    top_k = body.get("top_k", 3)

    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    print("Query.py PERPLEXITY_API_KEY (inside endpoint):", PERPLEXITY_API_KEY)
    print("Is PERPLEXITY_API_KEY falsy?", not PERPLEXITY_API_KEY)

    if not PERPLEXITY_API_KEY:
        raise HTTPException(status_code=500, detail="Missing Perplexity API key")

    # Step 1: Embed the question using CLIP
    inputs = clip_processor(text=question, return_tensors="pt", truncation=True, padding=True)
    question_embedding = clip_model.get_text_features(**inputs).detach().numpy()

    # Step 2: Search vector store
    index = get_faiss_index()
    context = search_index(index, question_embedding, k=top_k)

    # Step 3: If no context is found, return a fallback response
    if not context:
        return {
            "response": "No relevant documents found to answer the question.",
            "sources": [],
            "web_citations": []
        }

    # Step 4: Query LLM with RAG context
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = build_rag_prompt(question, context)
    
    payload = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.perplexity.ai/chat/completions", 
                json=payload, 
                headers=headers,
                timeout=30.0
            )
            print("Perplexity API Response:", response.text)
            response.raise_for_status()
            data = response.json()
            web_citations = data.get("citations", [])
            return {
                "response": data["choices"][0]["message"]["content"],
                "sources": [item[0]["source"] for item in context],
                "web_citations": web_citations
            }
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"API request failed: {e}")