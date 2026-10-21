import os
import tempfile
import pandas as pd
from docx import Document
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Your own functions from other files
from milvusload import extract_text_from_pdf, chunk_text, create_milvus_collection, insert_chunks_to_milvus
from milvussearch import search_similar_docs

# ===== Setup =====
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
st.set_page_config(page_title="Helpful Agent", page_icon="📄")
st.title("Helpful Chat Agent")

# ===== Helper to read different file types =====
def read_file(file):
    ext = file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(file.read())
        path = tmp.name

    if ext == "pdf":
        return extract_text_from_pdf(path)
    if ext == "docx":
        return "\n".join([p.text for p in Document(path).paragraphs])
    if ext == "txt":
        return open(path, encoding="utf-8").read()
    if ext == "csv":
        return pd.read_csv(path).to_string(index=False)
    return ""

# ===== File Upload =====
uploaded = st.file_uploader(
    "Upload a PDF, DOCX, TXT, or CSV",
    type=["pdf", "docx", "txt", "csv"],
    accept_multiple_files=False
)

if uploaded:
    text_data = read_file(uploaded)
    if text_data.strip():
        chunks = chunk_text(text_data)
        collection = create_milvus_collection()
        insert_chunks_to_milvus(chunks, collection)
        st.success("✅ File indexed successfully!")
    else:
        st.error("❌ Could not extract text from file.")

# ===== Question & Answer Section =====
answer_placeholder = st.empty()  # Answer will appear here
question = st.text_area("Your Question:")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        context = "\n\n".join(search_similar_docs(question, top_k=3))
        prompt = f"Answer based on the following context:\n{context}"
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=question)
        ]
        reply = llm.invoke(messages).content
        answer_placeholder.text_area("Answer:", value=reply, height=300, disabled=True)

