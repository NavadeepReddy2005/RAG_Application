from .RAG_Tool import Main
from langchain_core.tools import tool
from functools import lru_cache

@lru_cache(maxsize=1)
def get_rag():
    return Main.RAG()

@tool
def RAG_tools(query):
    """Search the document for information needed to answer the user's question.Data contains information about Hemanth and personal details about Hemanth."""

    return get_rag().invoke(query)