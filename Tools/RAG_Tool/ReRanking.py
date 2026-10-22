from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from config import Config

class ReRanker:

    def __init__(self):

        self.config = Config()

        self.base_retriever = self.config.Vector_db.as_retriver(k=5)

        self.Model = self.config.model

        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.base_retriever,
            llm=self.Model
        )

        self.ranker_model = HuggingFaceCrossEncoder(
            model_name = self.config.reranker_model_name
        )

        self.reranker = CrossEncoderReranker(
            model = self.ranker_model,
            top_n=3
        )

        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=self.multi_query_retriever
        )

    def rank(self,query):

        docs = self.compression_retriever.invoke(query)

        return docs
    