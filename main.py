from RAG_Tool import Main
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool


load_dotenv()

Model = ChatGroq(model='openai/gpt-oss-20b')

tools = Main.RAG()

@tool
def RAG_tools(query):
    """Search the document for information needed to answer the user's question.Data contains information about Hemanth and personal details about Hemanth."""
    rag = Main.RAG()

    return rag.invoke(query)

Model = Model.bind_tools([RAG_tools])

out = Model.invoke('how can i contact hemanth kumar')

for toolcall in out.tool_calls:

    if toolcall["name"] == "RAG_tools":
        response = RAG_tools.invoke(toolcall["args"])
        print("RAG Response:", response)

    else:
        print(out)