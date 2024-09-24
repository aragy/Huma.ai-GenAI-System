import re
import json
from typing import List, Tuple
from langchain.tools import DuckDuckGoSearchResults
from app.query_rewriter import QueryRewriter
from app.knowledge_refiner import KnowledgeRefiner

class WebSearcher:
    def __init__(self, llm):
        self.llm = llm
        self.search = DuckDuckGoSearchResults()

    def search_and_refine(self, query: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        rewritten_query = QueryRewriter(self.llm).rewrite(query)
        web_results = self.search.run(rewritten_query)
        web_knowledge = KnowledgeRefiner(self.llm).refine(web_results)
        sources = self.parse_search_results(web_results)
        return web_knowledge, sources

    @staticmethod
    def parse_search_results(results_string: str) -> List[Tuple[str, str]]:
        try:
            pattern = r'snippet: (.*?), title: (.*?), link: (https?://[^\s,]+)'
            matches = re.findall(pattern, results_string)
            return [(title, link) for snippet, title, link in matches]
        except Exception as e:
            print(f"Error parsing search results: {e}")
            return []
