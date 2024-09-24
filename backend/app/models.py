# backend/app/models.py

from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    model: str  

class QueryResponse(BaseModel):
    answer: str
