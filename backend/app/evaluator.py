from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

class RetrievalEvaluatorInput(BaseModel):
    relevance_score: float = Field(..., description="Relevance score between 0 and 1.")

class Evaluator:
    def __init__(self, llm):
        self.llm = llm

    def evaluate_relevance(self, query: str, document: str) -> float:
        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template=(
                "On a scale from 0 to 1, how relevant is the following document to the query?\n"
                "Query: {query}\n"
                "Document: {document}\n"
                "Relevance score:"
            )
        )
        chain = prompt | self.llm.with_structured_output(RetrievalEvaluatorInput)
        input_variables = {"query": query, "document": document}
        result = chain.invoke(input_variables).relevance_score
        return result
