from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

class QueryRewriterInput(BaseModel):
    query: str = Field(..., description="The rewritten query.")

class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, query: str) -> str:
        prompt = PromptTemplate(
            input_variables=["query"],
            template=(
                "Rewrite the following query to make it more suitable for a web search:\n"
                "{query}\n"
                "Rewritten query:"
            )
        )
        chain = prompt | self.llm.with_structured_output(QueryRewriterInput)
        input_variables = {"query": query}
        return chain.invoke(input_variables).query.strip()
