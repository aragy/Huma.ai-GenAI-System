import re
import json
from typing import List, Tuple
from langchain.tools import DuckDuckGoSearchResults
from app.query_rewriter import QueryRewriter
from app.knowledge_refiner import KnowledgeRefiner

class WebSearcher:
    """
    A web searcher that uses a language model (LLM) to search the web, refine search results, 
    and extract relevant information.

    This class performs a web search using the DuckDuckGo search engine, rewrites the query 
    for improved results, and refines the search output into key points. It also extracts the 
    source information (title and link) from the search results.

    Attributes:
        llm: A language model instance used to assist in query rewriting and refining results.
        search (DuckDuckGoSearchResults): An instance for retrieving search results from the web.
    """
    def __init__(self, llm):
        self.llm = llm
        self.search = DuckDuckGoSearchResults()

    def search_and_refine(self, query: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Performs a web search, rewrites the query, refines the search results, and extracts sources.

        Args:
            query (str): The original search query.

        Returns:
            Tuple[List[str], List[Tuple[str, str]]]: 
                - A list of key points extracted from the search results.
                - A list of tuples containing the title and link of each source.
        """
        rewritten_query = QueryRewriter(self.llm).rewrite(query)
        web_results = self.search.run(rewritten_query)
        web_knowledge = KnowledgeRefiner(self.llm).refine(web_results)
        sources = self.parse_search_results(web_results)
        return web_knowledge, sources

    @staticmethod
    def parse_search_results(results_string: str) -> List[Tuple[str, str]]:
        """
        Parses search results and extracts the title and link of each result.

        Args:
            results_string (str): The raw search results string.

        Returns:
            List[Tuple[str, str]]: A list of tuples where each tuple contains the title and link of a search result.
        """
        try:
            pattern = r'snippet: (.*?), title: (.*?), link: (https?://[^\s,]+)'
            matches = re.findall(pattern, results_string)
            return [(title, link) for snippet, title, link in matches]
        except Exception as e:
            print(f"Error parsing search results: {e}")
            return []
