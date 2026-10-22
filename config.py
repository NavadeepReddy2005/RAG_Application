from dataclasses import dataclass,field
from functools import lru_cache

from langchain_groq import ChatGroq
from Tools.RAG_Tool.VectorDB import VectorData

@lru_cache(maxsize=1)
def get_vector_db():

    return VectorData(
        collection_name="knowledge",
        directory="chroma"
    )

@dataclass
class Config:

    model_name: str = "openai/gpt-oss-20b"
    reasoning_effort: str = "low"

    reranker_model_name: str = "BAAI/bge-reranker-base"

    model:ChatGroq = field(init=False)
    Vector_db: VectorData = field(init=False)

    def __post_init__(self):

        self.model = ChatGroq(
            model=self.model_name,
            reasoning_effort=self.reasoning_effort
        )

        self.Vector_db = get_vector_db()

        