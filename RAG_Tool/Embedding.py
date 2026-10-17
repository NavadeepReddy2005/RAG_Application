from langchain_huggingface import HuggingFaceEmbeddings

class Embeddings():

    def __init__(self):

        self.embedding_model  = HuggingFaceEmbeddings(
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        )

    def embed_documents(self,texts):

        return self.embedding_model.embed_documents(texts)

    def embed_query(self,query):

        return self.embedding_model.embed_query(query)