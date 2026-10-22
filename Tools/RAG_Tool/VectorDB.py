from langchain_chroma import Chroma
from .Embedding import Embeddings

class VectorData:

    def  __init__(self,collection_name,directory):

        self.collection_name = collection_name

        self.directory = directory

        self.vectordb = Chroma(
            embedding_function = Embeddings(),
            collection_name = collection_name,
            persist_directory = directory
        )

    def add_docs(self,chunks):

        return self.vectordb.add_documents(chunks)

    def similarity_x(self,query,k=3):

        return self.vectordb.similarity_search(query,k)

    def remove_data(self):

        return self.vectordb.delete_collection()

    def as_retriver(self,k=3):

        return self.vectordb.as_retriever(
            search_kwargs = {'k':k}
        )