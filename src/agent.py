from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings

PROJECT_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_DIR / "faiss_index"
ENV_PATH = PROJECT_DIR / ".env"


def _environment_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def normalize_question(question: str) -> str:
    """Trim whitespace and reject empty input."""
    cleaned = (question or "").strip()
    return cleaned


def create_embeddings() -> AzureOpenAIEmbeddings | OpenAIEmbeddings:
    """Build the embeddings client from the project environment."""
    load_dotenv(ENV_PATH)

    azure_endpoint = _environment_value(
        "AZURE_OPENAI_EMBEDDING_ENDPOINT",
        "azure_embeddings_endpoint",
    )
    azure_api_key = _environment_value(
        "AZURE_OPENAI_EMBEDDING_API_KEY",
        "azure_embeddings_api_key",
    )
    deployment = _environment_value(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "azure_embeddings_deployment",
        default="text-embedding-ada-002",
    )

    if azure_endpoint and azure_api_key:
        arguments: dict[str, Any] = {
            "azure_endpoint": azure_endpoint,
            "api_key": azure_api_key,
            "azure_deployment": deployment,
        }
        api_version = _environment_value(
            "OPENAI_API_EMBEDDING_VERSION",
            "OPENAI_API_VERSION",
        )
        if api_version:
            arguments["openai_api_version"] = api_version
        return AzureOpenAIEmbeddings(**arguments)

    openai_api_key = _environment_value("OPENAI_API_KEY")
    if openai_api_key:
        return OpenAIEmbeddings(api_key=openai_api_key)

    raise RuntimeError(
        "Azure OpenAI embedding credentials or OPENAI_API_KEY were not found in the environment."
    )


def create_chat_model() -> ChatOpenAI:
    """Build the chat completion model used for grounded medical answers."""
    load_dotenv(ENV_PATH)

    endpoint = _environment_value("AZURE_OPENAI_ENDPOINT", "azure_endpoint")
    api_key = _environment_value("AZURE_OPENAI_API_KEY", "azure_api_key")
    deployment = _environment_value(
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "AZURE_OPENAI_DEPLOYMENT",
        "azure_deployment",
    )

    if endpoint and api_key and deployment:
        parsed_endpoint = urlsplit(endpoint)
        if not parsed_endpoint.scheme or not parsed_endpoint.netloc:
            raise RuntimeError("azure_endpoint must be a valid HTTPS endpoint URL.")
        base_url = f"{parsed_endpoint.scheme}://{parsed_endpoint.netloc}/openai/v1/"
        return ChatOpenAI(
            model=deployment,
            base_url=base_url,
            api_key=api_key,
            temperature=0,
        )

    openai_api_key = _environment_value("OPENAI_API_KEY")
    if openai_api_key:
        return ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0)

    raise RuntimeError(
        "Set azure_endpoint, azure_api_key, and azure_deployment in .env or provide OPENAI_API_KEY."
    )


def load_vectorstore(index_dir: str | Path = INDEX_DIR) -> FAISS:
    """Load the local FAISS index generated from the medical dataset."""
    index_path = Path(index_dir)
    index_file = index_path / "index.faiss"
    metadata_file = index_path / "index.pkl"

    if not index_file.is_file() or not metadata_file.is_file():
        raise FileNotFoundError(
            f"Expected FAISS files in {index_path}. Run the ingestion step first."
        )

    return FAISS.load_local(
        str(index_path),
        create_embeddings(),
        index_name="index",
        allow_dangerous_deserialization=True,
    )


def retrieve_context(
    question: str,
    vectorstore: FAISS | None = None,
    top_k: int = 4,
) -> list[Document]:
    """Fetch the relevant medical passages for a question."""
    normalized = normalize_question(question)
    if not normalized:
        raise ValueError("Question cannot be empty.")

    store = vectorstore if vectorstore is not None else load_vectorstore()
    return store.similarity_search(normalized, k=top_k)


def build_prompt(question: str, contexts: Sequence[Document]) -> str:
    """Construct a grounded medical answer prompt."""
    if not contexts:
        return (
            "You are a medical information assistant. The retrieved context is empty, "
            "so say that the indexed materials do not contain enough information to answer this question. "
            "Do not diagnose or prescribe. Encourage the user to consult a qualified healthcare professional.\n\n"
            f"Question: {question}"
        )

    context_text = "\n\n".join(document.page_content for document in contexts)
    return (
        "You are a medical information assistant. Answer using only the context below. "
        "If the context is insufficient, say so clearly. Do not diagnose, prescribe, or provide treatment plans. "
        "Encourage the user to seek advice from a qualified healthcare professional for personal medical decisions.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}"
    )


def answer_question(
    question: str,
    vectorstore: FAISS | None = None,
    top_k: int = 4,
) -> tuple[str, list[Document]]:
    """Answer a medical question using the FAISS knowledge base."""
    normalized = normalize_question(question)
    if not normalized:
        raise ValueError("Enter a question before submitting.")

    documents = retrieve_context(normalized, vectorstore=vectorstore, top_k=top_k)
    prompt = build_prompt(normalized, documents)
    response = create_chat_model().invoke(prompt)
    content = response.content
    answer = content if isinstance(content, str) else "\n".join(str(part) for part in content)
    return answer, list(documents)


class MedicalRAGAgent:
    """A compact wrapper for the medical RAG pipeline."""

    def __init__(self, vectorstore: FAISS | None = None, top_k: int = 4):
        self.vectorstore = vectorstore
        self.top_k = top_k

    def ask(self, question: str) -> tuple[str, list[Document]]:
        return answer_question(question, vectorstore=self.vectorstore, top_k=self.top_k)

    def __call__(self, question: str) -> tuple[str, list[Document]]:
        return self.ask(question)


def build_agent(vectorstore: FAISS | None = None, top_k: int = 4) -> MedicalRAGAgent:
    """Convenience constructor for the project agent."""
    return MedicalRAGAgent(vectorstore=vectorstore, top_k=top_k)


if __name__ == "__main__":
    print("Medical RAG agent ready. Type 'exit' to quit.")
    agent = build_agent()
    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        try:
            answer, documents = agent.ask(question)
            print(f"\n{answer}\n")
            if documents:
                print("Sources:")
                for idx, doc in enumerate(documents, start=1):
                    source = doc.metadata.get("source", "Saved FAISS document")
                    print(f"  [{idx}] {source}")
        except Exception as error:
            print(f"Error: {error}")
