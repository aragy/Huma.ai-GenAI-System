from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

class QueryRewriterInput(BaseModel):
    query: str = Field(..., description="The rewritten query.")

class QueryRewriter:
    """
    A class that rewrites user queries to make them more suitable for web searches using a language model (LLM).

    This class takes a query and reformulates it to improve its clarity and relevance for search engines, making
    the query more effective in retrieving accurate results.

    Attributes:
        llm: A language model instance used to perform query rewriting.
    """
    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, query: str) -> str:
        """
        Rewrites a query to make it more suitable for web searches.

        Args:
            query (str): The original query string to be rewritten.

        Returns:
            str: The rewritten query, formatted for better suitability in web searches.
        """
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
