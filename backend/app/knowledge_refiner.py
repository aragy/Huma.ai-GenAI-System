from typing import List
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

class KnowledgeRefinementInput(BaseModel):
    key_points: str = Field(..., description="Extracted key points.")

class KnowledgeRefiner:
    def __init__(self, llm):
        self.llm = llm

    def refine(self, document: str) -> List[str]:
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
