import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
import time

load_dotenv()

azure_embeddings_api_key = os.getenv("azure_embeddings_api_key")
azure_embeddings_endpoint = os.getenv("azure_embeddings_endpoint")
azure_embeddings_deployment = os.getenv("azure_embeddings_deployment")

azure_api_key = os.getenv("azure_api_key")
azure_endpoint = os.getenv("azure_endpoint")
azure_deployment = os.getenv("azure_deployment")


# Creating RAG Pipeline

def load_pipeline():
    #Create OPENAI Embeddings

    openai_embeddings = OpenAI(
                    api_key = azure_embeddings_api_key,
                    base_url = azure_embeddings_endpoint
                    azure_deployment = azure_embeddings_deployment)


    

    




