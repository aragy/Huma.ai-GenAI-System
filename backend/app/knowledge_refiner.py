from typing import List
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

class KnowledgeRefinementInput(BaseModel):
    key_points: str = Field(..., description="Extracted key points.")

class KnowledgeRefiner:
    """
    A class that refines knowledge by extracting key information from a document using a language model (LLM).

    This class processes a document to identify the most important points and returns them in a structured
    format, such as a list of bullet points.

    Attributes:
        llm: A language model instance used to perform knowledge refinement.
    """
    def __init__(self, llm):
        self.llm = llm

    def refine(self, document: str) -> List[str]:
        """
        Extracts the key information from a document in the form of bullet points.

        Args:
            document (str): The document from which to extract key points.

        Returns:
            List[str]: A list of bullet points containing the key information from the document.
        """
        prompt = PromptTemplate(
            input_variables=["document"],
            template=(
                "Extract the key information from the following document in bullet points:\n"
                "{document}\n"
                "Key points:"
            )
        )
        chain = prompt | self.llm.with_structured_output(KnowledgeRefinementInput)
        input_variables = {"document": document}
        result = chain.invoke(input_variables).key_points
        return [point.strip('- ').strip() for point in result.split('\n') if point.strip()]
