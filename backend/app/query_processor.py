from typing import List, Tuple
from app.retriever import DocumentRetriever
from app.evaluator import Evaluator
from app.web_searcher import WebSearcher
from app.knowledge_refiner import KnowledgeRefiner
from app.response_generator import ResponseGenerator

class QueryProcessor:
    def __init__(self, retriever: DocumentRetriever, evaluator: Evaluator, web_searcher: WebSearcher, llm):
        self.retriever = retriever
        self.evaluator = evaluator
        self.web_searcher = web_searcher
        self.llm = llm

    def process(self, query: str, eval_documents: bool = True) -> str:
        """
        Process the query and generate a response.

        Args:
            query (str): The query string.
            eval_documents (bool): Whether to evaluate documents' relevance.

        Returns:
            str: The generated response.
        """
        retrieved_docs = self.retriever.retrieve(query)
        
        if eval_documents:
            eval_scores = [
                self.evaluator.evaluate_relevance(query, doc.page_content)
                for doc in retrieved_docs
            ]
            max_score = max(eval_scores)

            if max_score > 0.7:
                best_doc = retrieved_docs[eval_scores.index(max_score)]
                final_knowledge = best_doc.page_content
                sources = [("Retrieved Document", "")]
            elif max_score < 0.3:

                web_knowledge, sources = self.web_searcher.search_and_refine(query)
                final_knowledge = "\n".join(web_knowledge)
            else:
                best_doc = retrieved_docs[eval_scores.index(max_score)]
                retrieved_knowledge = KnowledgeRefiner(self.llm).refine(best_doc.page_content)
                web_knowledge, web_sources = self.web_searcher.search_and_refine(query)
                final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
                sources = [("Retrieved Document", "")] + web_sources
        else:
            final_knowledge = retrieved_docs[0].page_content
            sources = [("Retrieved Document", "")]

        response = ResponseGenerator(self.llm).generate(query, final_knowledge, sources)
        return response
