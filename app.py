import streamlit as st
import json
import os

from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="📚",
    layout="centered"
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {
    padding-top: 2rem;
    max-width: 800px;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Knowledge Assistant")
st.caption("Chat with documents, generate quizzes, and study smarter")


# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("📚 AI Knowledge Assistant")
st.sidebar.caption("Built by Kartik Vagh")

mode = st.sidebar.selectbox(
    "Study Mode",
    [
        "Ask Questions",
        "Explain Simply",
        "Generate Quiz",
        "Create Flashcards"
    ]
)

menu = st.sidebar.radio(
    "Navigation",
    ["💬 Chat", "📂 Upload Documents"]
)


# -----------------------------
# API KEY
# -----------------------------

api_key = st.secrets["OPENAI_API_KEY"]


# -----------------------------
# LOAD USERS
# -----------------------------

if not os.path.exists("users.json"):
    with open("users.json", "w") as f:
        json.dump({}, f)

with open("users.json", "r") as f:
    users = json.load(f)


# -----------------------------
# LOGIN SYSTEM
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
# AFTER LOGIN
# -----------------------------

username = st.session_state.username

st.sidebar.markdown("---")
st.sidebar.write(f"👤 {users[username]['name']}")
st.sidebar.markdown("---")


# -----------------------------
# SESSION CONTROLS
# -----------------------------

if st.sidebar.button("🆕 New Session"):
    st.session_state.retrievers = None
    st.session_state.messages = []
    st.rerun()

if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()


# -----------------------------
# SESSION STATE
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrievers" not in st.session_state:
    st.session_state.retrievers = None


# -----------------------------
# LOAD LLM
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
# QUERY EXPANSION
# -----------------------------

def expand_queries(question):

    prompt = f"""
Generate 4 alternative search queries for the question.
Return each query on a new line.

Question: {question}
"""

    response = llm.invoke(prompt).content

    queries = [q.strip() for q in response.split("\n") if q.strip()]
    queries.append(question)

    return queries


# -----------------------------
# KEYWORD FILTER
# -----------------------------

def keyword_match_filter(docs, question):

    words = question.lower().split()

    filtered = []

    for doc in docs:
        text = doc.page_content.lower()

        if any(word in text for word in words):
            filtered.append(doc)

    return filtered


# -----------------------------
# DOCUMENT UPLOAD
# -----------------------------

if menu == "📂 Upload Documents":

    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True
    )

    MAX_DOCS = 5

    if uploaded_files and len(uploaded_files) > MAX_DOCS:
        st.error("Maximum 5 documents allowed per session.")
        st.stop()

    if uploaded_files and st.session_state.retrievers is None:

        with st.spinner("Analyzing documents..."):

            all_docs = []

            for uploaded_file in uploaded_files:

                if uploaded_file.size > 20 * 1024 * 1024:
                    st.error("File too large (max 20MB).")
                    st.stop()

                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.read())

                loader = PyPDFLoader(uploaded_file.name)
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name

                all_docs.extend(docs)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200
            )

            split_docs = splitter.split_documents(all_docs)

            embeddings = OpenAIEmbeddings(openai_api_key=api_key)

            vector_db = FAISS.from_documents(split_docs, embeddings)

            vector_retriever = vector_db.as_retriever(search_kwargs={"k": 20})

            bm25_retriever = BM25Retriever.from_documents(split_docs)
            bm25_retriever.k = 20

            st.session_state.retrievers = {
                "vector": vector_retriever,
                "bm25": bm25_retriever
            }

        st.success("Documents indexed successfully!")


# -----------------------------
# CHAT INTERFACE
# -----------------------------

if menu == "💬 Chat":

    st.header("Chat with Documents")

    if st.session_state.retrievers is None:
        st.info("Upload documents first.")
        st.stop()

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask something about the documents")

    if question:

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        queries = expand_queries(question)

        relevant_docs = []

        for q in queries:

            vector_docs = st.session_state.retrievers["vector"].invoke(q)
            keyword_docs = st.session_state.retrievers["bm25"].invoke(q)

            relevant_docs.extend(vector_docs)
            relevant_docs.extend(keyword_docs)

        unique_docs = []
        seen = set()

        for doc in relevant_docs:

            text = doc.page_content

            if text not in seen:
                unique_docs.append(doc)
                seen.add(text)

        relevant_docs = keyword_match_filter(unique_docs, question)

        MAX_CONTEXT_DOCS = 6
        selected_docs = relevant_docs[:MAX_CONTEXT_DOCS]

        context = "\n\n".join(
            [doc.page_content for doc in selected_docs]
        )

        # -----------------------------
        # MODE PROMPTS
        # -----------------------------

        if mode == "Ask Questions":

            prompt = f"""
Answer the question using the document context.

Context:
{context}

Question:
{question}

Provide a clear answer.
"""

        elif mode == "Explain Simply":

            prompt = f"""
Explain the following content in simple language
so a student can understand easily.

Document:
{context}
"""

        elif mode == "Generate Quiz":

            prompt = f"""
Create 5 quiz questions from the document.

Context:
{context}

Return numbered questions.
"""

        elif mode == "Create Flashcards":

            prompt = f"""
Create 10 flashcards.

Format:
Q: question
A: answer

Context:
{context}
"""

        try:
            response = llm.invoke(prompt).content

        except Exception:
            response = "AI service temporarily unavailable."

        with st.chat_message("assistant"):

            st.markdown(response)

            st.markdown("### Sources")

            shown = set()

            for doc in selected_docs:

                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "?")

                key = f"{source}-{page}"

                if key not in shown:
                    st.markdown(f"- {source} (Page {page})")
                    shown.add(key)

            with st.expander("📄 View Source Text"):

                for doc in selected_docs:

                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "?")

                    st.markdown(f"**{source} – Page {page}**")
                    st.write(doc.page_content[:800])

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )


# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")
st.markdown("© 2026 Kartik Vagh")
