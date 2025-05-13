# app/schemas.py
from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
