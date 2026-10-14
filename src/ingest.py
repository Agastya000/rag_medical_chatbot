# 1. Import Library
from langchain_community.document_loaders import JSONLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from openai import AzureOpenAI
import os
 
load_dotenv()

azure_embeddings_api_key = os.getenv("azure_embeddings_api_key")
azure_embeddings_endpoint = os.getenv("azure_embeddings_endpoint")
azure_embeddings_deployment = os.getenv("azure_embeddings_deployment")

#2 Load The DataSet




# 3. Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=100,
                                               separators=["\n\n", "\n", " ", ""])


# 4. Create embeddings and store them in FAISS vectorstore
from langchain_openai import OpenAIEmbeddings
openai_embeddings = OpenAI(
                    api_key = azure_embeddings_api_key,
                    base_url = azure_embeddings_endpoint
                    azure_deployment = azure_embeddings_deployment
)
