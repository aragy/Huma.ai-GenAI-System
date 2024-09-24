from typing import List
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

class DocumentRetriever:
    def __init__(self, bm25_docs: List[str], faiss_docs: List[str]):
        self.bm25_retriever = BM25Retriever.from_texts(
            bm25_docs, metadatas=[{"source": "BM25"}] * len(bm25_docs)
        )
        self.bm25_retriever.k = 5

        embedding = OpenAIEmbeddings()
        faiss_vectorstore = FAISS.from_texts(
            faiss_docs, embedding, metadatas=[{"source": "FAISS"}] * len(faiss_docs)
        )
        self.faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever], weights=[0.5, 0.5]
        )

    def retrieve(self, query: str) -> List[Document]:
        return self.ensemble_retriever.invoke(query)
