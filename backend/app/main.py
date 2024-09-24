from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List
import os

from app.query_processor import QueryProcessor
from app.models import QueryRequest, QueryResponse
from app.retriever import DocumentRetriever
from app.evaluator import Evaluator
from app.web_searcher import WebSearcher
from app.utils import load_documents
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI()


documents = load_documents('data/cleaned_dataset.csv')
retriever = DocumentRetriever(documents, documents)
load_dotenv() 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")

llms = {
    'gpt': ChatOpenAI(
        model="gpt-4",
        max_tokens=1000,
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    ),
    'claude': ChatAnthropic(
        temperature=0,
        model_name='claude-2',
        anthropic_api_key=ANTHROPIC_API_KEY
    ),
    'gemini': ChatGoogleGenerativeAI(
        model="gemini-1.5",
        temperature=0,
        google_genai_api_key=GOOGLE_GENAI_API_KEY
    )
}

processors = {}
for model_name, llm in llms.items():
    evaluator = Evaluator(llm)
    web_searcher = WebSearcher(llm)
    processors[model_name] = QueryProcessor(retriever, evaluator, web_searcher, llm)

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    model_name = request.model.lower()
    query = request.query

    if model_name not in processors:
        raise HTTPException(status_code=400, detail="Invalid model selected. Choose from 'gpt', 'claude', or 'gemini'.")

    processor = processors[model_name]
    response_text = processor.process(query)

    return QueryResponse(answer=response_text)
