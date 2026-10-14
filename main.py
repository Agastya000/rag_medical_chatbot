from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, AzureOpenAIEmbeddings


PROJECT_DIR = Path(__file__).resolve().parent
INDEX_DIR = PROJECT_DIR / "faiss_index"


def _environment_value(*names: str, default: str | None = None) -> str | None:
	for name in names:
		value = os.getenv(name)
		if value:
			return value
	return default


def normalize_question(question: str) -> str:
	question = question.strip()
	if not question or '"' not in question:
		return question

	try:
		fields = next(csv.reader([question], skipinitialspace=True))
	except csv.Error:
		return question

	if len(fields) == 2 and fields[0].strip().endswith("?") and fields[1].strip():
		return fields[0].strip()
	return question


def create_embeddings() -> AzureOpenAIEmbeddings:
	load_dotenv(PROJECT_DIR / ".env")
	endpoint = _environment_value(
		"AZURE_OPENAI_EMBEDDING_ENDPOINT", "azure_embeddings_endpoint"
	)
	api_key = _environment_value(
		"AZURE_OPENAI_EMBEDDING_API_KEY", "azure_embeddings_api_key"
	)
	deployment = _environment_value(
		"AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
		"azure_embeddings_deployment",
		default="text-embedding-ada-002",
	)
	if not endpoint or not api_key:
		raise RuntimeError(
			"Set azure_embeddings_endpoint and azure_embeddings_api_key in .env."
		)

	arguments: dict[str, Any] = {
		"azure_endpoint": endpoint,
		"api_key": api_key,
		"azure_deployment": deployment,
	}
	api_version = _environment_value(
		"OPENAI_API_EMBEDDING_VERSION", "OPENAI_API_VERSION"
	)
	if api_version:
		arguments["openai_api_version"] = api_version
	return AzureOpenAIEmbeddings(**arguments)


def load_vectorstore() -> FAISS:
	index_file = INDEX_DIR / "index.faiss"
	metadata_file = INDEX_DIR / "index.pkl"
	if not index_file.is_file() or not metadata_file.is_file():
		raise FileNotFoundError(
			f"Expected index.faiss and index.pkl in {INDEX_DIR}."
		)

	return FAISS.load_local(
		str(INDEX_DIR),
		create_embeddings(),
		index_name="index",
		allow_dangerous_deserialization=True,
	)


def create_chat_model() -> ChatOpenAI:
	load_dotenv(PROJECT_DIR / ".env")
	endpoint = _environment_value("AZURE_OPENAI_ENDPOINT", "azure_endpoint")
	api_key = _environment_value("AZURE_OPENAI_API_KEY", "azure_api_key")
	deployment = _environment_value(
		"AZURE_OPENAI_CHAT_DEPLOYMENT",
		"AZURE_OPENAI_DEPLOYMENT",
		"azure_deployment",
	)
	if not endpoint or not api_key or not deployment:
		raise RuntimeError(
			"Set azure_endpoint, azure_api_key, and azure_deployment in .env."
		)

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


def answer_question(
	question: str,
	vectorstore: FAISS | None = None,
	top_k: int = 4,
) -> tuple[str, list[Document]]:
	question = normalize_question(question)
	if not question:
		raise ValueError("Enter a question before submitting.")

	store = vectorstore if vectorstore is not None else load_vectorstore()
	documents = store.similarity_search(question, k=top_k)
	context = "\n\n".join(document.page_content for document in documents)
	prompt = (
		"You are a medical information assistant. Answer using only the context below. "
		"If the context does not contain the answer, say so. Do not diagnose or prescribe; "
		"encourage consulting a qualified healthcare professional for personal advice.\n\n"
		f"Context:\n{context}\n\nQuestion: {question}"
	)
	response = create_chat_model().invoke(prompt)
	content = response.content
	if isinstance(content, str):
		answer = content
	else:
		answer = "\n".join(str(part) for part in content)
	return answer, documents


def _print_sources(documents: list[Document]) -> None:
	for number, document in enumerate(documents, start=1):
		source = document.metadata.get("source", "Saved FAISS document")
		row = document.metadata.get("row")
		suffix = f", row {row}" if row is not None else ""
		print(f"[{number}] {source}{suffix}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Ask the medical FAISS knowledge base.")
	parser.add_argument("question", nargs="?", help="Question to ask")
	args = parser.parse_args()

	if args.question:
		answer, documents = answer_question(args.question)
		print(answer)
		print("\nSources:")
		_print_sources(documents)
		return

	print("Medical knowledge base. Enter 'quit' to exit.")
	while True:
		question = input("\nQuestion: ").strip()
		if question.lower() in {"quit", "exit"}:
			break
		if not question:
			continue
		try:
			answer, documents = answer_question(question)
			print(f"\n{answer}\n\nSources:")
			_print_sources(documents)
		except Exception as error:
			print(f"Error: {error}")


if __name__ == "__main__":
	main()
