from typing import List, Tuple
from langchain.prompts import PromptTemplate

class ResponseGenerator:
    """
    A class that generates a response based on the query, extracted knowledge, and sources using a language model (LLM).

    This class takes the query, the relevant knowledge extracted from the web search, and a list of sources
    to generate a comprehensive response. The sources are appended to the end of the response, including
    their links where available.

    Attributes:
        llm: A language model instance used to generate the response.
    """
    def __init__(self, llm):
        self.llm = llm

    def generate(self, query: str, knowledge: str, sources: List[Tuple[str, str]]) -> str:
        """
        Generates a response to the query based on the provided knowledge and sources.

        Args:
            query (str): The query string for which the response is generated.
            knowledge (str): The knowledge extracted from relevant documents.
            sources (List[Tuple[str, str]]): A list of tuples containing the title and link of sources.

        Returns:
            str: A generated response that answers the query, including references to the sources.
        """
        sources_formatted = "\n".join(
            [f"- {title}: {link}" if link else f"- {title}" for title, link in sources]
        )
        prompt = PromptTemplate(
            input_variables=["query", "knowledge", "sources"],
            template=(
                "Based on the following knowledge, answer the query. "
                "Include the sources with their links (if available) at the end of your answer:\n"
                "Query: {query}\n"
                "Knowledge: {knowledge}\n"
                "Sources:\n{sources}\n"
                "Answer:"
            )
        )
        input_variables = {
            "query": query,
            "knowledge": knowledge,
            "sources": sources_formatted
        }
        chain = prompt | self.llm
        return chain.invoke(input_variables).content.strip()
