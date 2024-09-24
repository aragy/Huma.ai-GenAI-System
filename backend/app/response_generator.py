from typing import List, Tuple
from langchain.prompts import PromptTemplate

class ResponseGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, query: str, knowledge: str, sources: List[Tuple[str, str]]) -> str:
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
