import streamlit as st
import json
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="AI Document Assistant", page_icon="📚")

st.title("📚 AI Knowledge Assistant")
st.caption("Built by Kartik Vagh")

api_key = st.secrets["OPENAI_API_KEY"]

# -----------------------------
# Load users
# -----------------------------

with open("users.json") as f:
    users = json.load(f)

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if username not in users or users[username]["password"] != password:
    st.stop()

st.success(f"Welcome {users[username]['name']}")

# -----------------------------
# User folders
# -----------------------------

user_folder = f"data/{username}"
docs_folder = f"{user_folder}/docs"
db_folder = f"{user_folder}/vectordb"

os.makedirs(docs_folder, exist_ok=True)
os.makedirs(db_folder, exist_ok=True)

# -----------------------------
# Session state
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

# -----------------------------
# Load LLM
# -----------------------------

@st.cache_resource
def load_llm():

    return ChatOpenAI(
        openai_api_key=api_key,
        temperature=0
    )

# -----------------------------
# Create DB
# -----------------------------

def create_db():

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    all_docs = []

    for file in os.listdir(docs_folder):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(f"{docs_folder}/{file}")

            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file

            all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(all_docs)

    db = FAISS.from_documents(split_docs, embeddings)

    db.save_local(db_folder)

    return db

# -----------------------------
# Load DB if exists
# -----------------------------

def load_db():

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    if os.path.exists(db_folder):

        return FAISS.load_local(db_folder, embeddings, allow_dangerous_deserialization=True)

    return None

if st.session_state.db is None:

    st.session_state.db = load_db()

# -----------------------------
# Upload docs
# -----------------------------

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    for uploaded_file in uploaded_files:

        path = f"{docs_folder}/{uploaded_file.name}"

        with open(path, "wb") as f:
            f.write(uploaded_file.read())

    st.session_state.db = create_db()

    st.success("Documents indexed successfully!")

# -----------------------------
# Chat
# -----------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

question = st.chat_input("Ask something about your documents")

if question and st.session_state.db:

    llm = load_llm()

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):

        st.markdown(question)

    results = st.session_state.db.similarity_search_with_score(question)

    relevant_docs = []

    for doc, score in results:

        if score < 0.7:
            relevant_docs.append(doc)

    if len(relevant_docs) == 0:

        response = "I could not find that in the documents."

    else:

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
You are a document assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{question}
"""

        response = llm.predict(prompt)

    with st.chat_message("assistant"):

        st.markdown(response)

        st.markdown("**Sources:**")

        shown = set()

        for doc in relevant_docs:

            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")

            key = f"{source}-{page}"

            if key not in shown:

                st.markdown(f"- {source} (Page {page})")

                shown.add(key)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

st.markdown("---")
st.markdown("© 2026 Kartik Vagh")
