from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

class RetrievalEvaluatorInput(BaseModel):
    relevance_score: float = Field(..., description="Relevance score between 0 and 1.")

class Evaluator:
    """
    An evaluator that measures the relevance of a document to a query using a language model (LLM).

    This class uses a language model to provide a relevance score, on a scale from 0 to 1, 
    indicating how relevant a given document is to a specific query. 

    Attributes:
        llm: A language model instance that is used to evaluate relevance.
    """
    def __init__(self, llm):
        self.llm = llm

    def evaluate_relevance(self, query: str, document: str) -> float:
        """
        Evaluates the relevance of a document to a query using the LLM.

        Args:
            query (str): The query string to be matched.
            document (str): The document string to be evaluated for relevance.

        Returns:
            float: A relevance score between 0 and 1, with 1 being the highest relevance.
        """
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
