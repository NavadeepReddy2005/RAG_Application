from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DataLoader:

    def __init__(self):
        pass

    def Load_Data(self,path):

        Loader = PyMuPDFLoader(path)

        return Loader.load()

    def Split_Chunks(self,data):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size  = 400,
            chunk_overlap = 100,
            separators = ['\n','\n\n','.'],
            keep_separator = False
        )

        chunks = splitter.split_documents(data)

        return chunks
    