from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from main import answer_question, load_vectorstore, normalize_question


load_dotenv(Path(__file__).resolve().parent / ".env")

st.set_page_config(page_title="Medical Knowledge Assistant")
st.title("Medical Knowledge Assistant")
st.caption("Answers are grounded in the saved medical document index.")


@st.cache_resource(show_spinner="Loading the medical index...")
def get_vectorstore():
	return load_vectorstore()


if "messages" not in st.session_state:
	st.session_state.messages = []

for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.markdown(message["content"])
		if message.get("sources"):
			with st.expander("Retrieved sources"):
				for number, source in enumerate(message["sources"], start=1):
					st.markdown(f"**[{number}] {source['name']}**")
					st.caption(source["excerpt"])

question = st.chat_input("Ask a question about the indexed medical data")
if question:
	question = normalize_question(question)
	st.session_state.messages.append({"role": "user", "content": question})
	with st.chat_message("user"):
		st.markdown(question)

	with st.chat_message("assistant"):
		try:
			with st.spinner("Searching the index and preparing an answer..."):
				answer, documents = answer_question(
					question,
					vectorstore=get_vectorstore(),
				)
			st.markdown(answer)
			sources = [
				{
					"name": str(document.metadata.get("source", "Saved FAISS document")),
					"excerpt": document.page_content[:800],
				}
				for document in documents
			]
			if sources:
				with st.expander("Retrieved sources"):
					for number, source in enumerate(sources, start=1):
						st.markdown(f"**[{number}] {source['name']}**")
						st.caption(source["excerpt"])
			st.session_state.messages.append(
				{"role": "assistant", "content": answer, "sources": sources}
			)
		except Exception as error:
			error_message = str(error)
			st.error(error_message)
			st.session_state.messages.append(
				{"role": "assistant", "content": f"Unable to answer: {error_message}"}
			)
