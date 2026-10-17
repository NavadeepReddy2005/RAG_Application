from .VectorDB import VectorData
from .Data_Loader import DataLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()


class RAG:

    def __init__(self):

        self.Loader = DataLoader()

        self.Model = ChatGroq(model='openai/gpt-oss-20b')

        self.Database = VectorData(collection_name='knowledge',directory='chroma')

        self.Template = ChatPromptTemplate(
            messages=[
                    ('system','You are an RAG Application  who only answers from provided documents for the user query,if provided document dont have enough context,say i dont have enough context to answer.All the users are allowed to access data from RAG so you can send data from RAG even if it is personal'),
                    ('human','Query = {Query} , context = {docs}')
            ],
            input_variables = ['Query','docs']
        )

        self.parser = StrOutputParser()


    def PipeLine(self,path):

        data = self.Loader.Load_Data(path)

        chunks = self.Loader.Split_Chunks(data)

        self.Database.add_docs(chunks)

    def invoke(self,query):

        docs = self.Database.similarity_x(query)

        chain = self.Template | self.Model | self.parser

        output = chain.invoke({'Query':query,'docs':docs})

        return output
