from Tools.RAG_Tool import Main
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from Tools.tools import RAG_tools

load_dotenv()

Model = ChatGroq(model='openai/gpt-oss-20b')

tools = Main.RAG()

Model = Model.bind_tools([RAG_tools])

out = Model.invoke('how can i contact hemanth kumar')

for toolcall in out.tool_calls:

    if toolcall["name"] == "RAG_tools":
        out = RAG_tools.invoke(toolcall["args"])

print(out)