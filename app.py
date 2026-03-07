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

api_key = st.secrets["OPENAI_API_KEY"]


# -----------------------------
# USER SYSTEM
# -----------------------------

if not os.path.exists("users.json"):
    with open("users.json", "w") as f:
        json.dump({}, f)

with open("users.json", "r") as f:
    users = json.load(f)


# -----------------------------
# LOGIN
# -----------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username in users and users[username]["password"] == password:

            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()

        else:
            st.error("Invalid username or password")

    st.stop()


username = st.session_state.username

st.sidebar.markdown("---")
st.sidebar.write(f"👤 {users[username]['name']}")
st.sidebar.markdown("---")


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
# RERANK FUNCTION
# -----------------------------

def rerank_docs(docs, question):

    words = question.lower().split()

    scored = []

    for doc in docs:

        text = doc.page_content.lower()

        score = sum(word in text for word in words)

        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [doc for score, doc in scored]


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
# DOCUMENT UPLOAD
# -----------------------------

if menu == "📂 Upload Documents":

    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files and st.session_state.retrievers is None:

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
# CHAT
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

        retrieved_docs = []

        for q in queries:

            vector_docs = st.session_state.retrievers["vector"].invoke(q)
            keyword_docs = st.session_state.retrievers["bm25"].invoke(q)

            retrieved_docs.extend(vector_docs)
            retrieved_docs.extend(keyword_docs)

        unique_docs = []
        seen = set()

        for doc in retrieved_docs:

            text = doc.page_content

            if text not in seen:
                unique_docs.append(doc)
                seen.add(text)

        ranked_docs = rerank_docs(unique_docs, question)

        selected_docs = ranked_docs[:4]

        if len(selected_docs) == 0:

            response = "No relevant information found in the document."

        else:

            context = "\n\n".join(
                [doc.page_content for doc in selected_docs]
            )

            if mode == "Ask Questions":

                prompt = f"""
Answer ONLY using the document context.

If the answer is not present say:
"Not found in the document."

Context:
{context}

Question:
{question}
"""

            elif mode == "Explain Simply":

                prompt = f"""
Explain the answer to the user's question
in simple language.

Only use the document context.

Context:
{context}

Question:
{question}
"""

            elif mode == "Generate Quiz":

                prompt = f"""
Create 5 quiz questions from the document.

Use ONLY information in the context.

Context:
{context}
"""

            elif mode == "Create Flashcards":

                prompt = f"""
Create 5 flashcards using ONLY the document.

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

            if len(selected_docs) > 0:

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
