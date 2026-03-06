import streamlit as st
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain.retrievers.multi_query import MultiQueryRetriever

st.set_page_config(page_title="AI Knowledge Assistant", page_icon="📚")

st.title("📚 AI Knowledge Assistant")
st.caption("Built by Kartik Vagh")

api_key = st.secrets["OPENAI_API_KEY"]

# -----------------------------
# Load users
# -----------------------------

with open("users.json") as f:
    users = json.load(f)

# -----------------------------
# Login system
# -----------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    login = st.button("Login")

    if login:

        if username in users and users[username]["password"] == password:

            st.session_state.logged_in = True
            st.session_state.username = username

            st.rerun()

        else:
            st.error("Invalid username or password")

    st.stop()

# -----------------------------
# After login
# -----------------------------

username = st.session_state.username

st.success(f"Welcome {users[username]['name']}")

# -----------------------------
# Logout button
# -----------------------------

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

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
        model="gpt-4.1-mini",
        openai_api_key=api_key,
        temperature=0
    )

llm = load_llm()

# -----------------------------
# Upload documents
# -----------------------------

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    with st.spinner("Analyzing documents..."):

        all_docs = []

        for uploaded_file in uploaded_files:

            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.read())

            loader = PyPDFLoader(uploaded_file.name)

            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = uploaded_file.name

            all_docs.extend(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        split_docs = splitter.split_documents(all_docs)

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        st.session_state.db = FAISS.from_documents(split_docs, embeddings)

        st.session_state.messages = []

    st.success("Documents indexed successfully!")

# -----------------------------
# Display chat history
# -----------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# -----------------------------
# Chat input
# -----------------------------

question = st.chat_input("Ask something about the documents")

if question and st.session_state.db:

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):

        st.markdown(question)

    # Multi-query retrieval
    retriever = MultiQueryRetriever.from_llm(
        retriever=st.session_state.db.as_retriever(
            search_kwargs={"k": 20}
        ),
        llm=llm
    )

    relevant_docs = retriever.get_relevant_documents(question)

    if len(relevant_docs) == 0:

        response = "I could not find that in the documents."

    else:

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
You are an AI document assistant.

Answer ONLY using the provided context.

Rules:
- If the answer is not in the documents, say you cannot find it.
- Do not invent information.

Context:
{context}

Question:
{question}
"""

        response = llm.invoke(prompt).content

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
