import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Document Chatbot", page_icon="📄")

st.title("📄 Document Chatbot")
st.caption("Built by Kartik Vagh")

# -------------------
# API Key
# -------------------

api_key = st.text_input("Enter OpenAI API Key", type="password")

# -------------------
# Session State
# -------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

# -------------------
# Cached LLM
# -------------------

@st.cache_resource
def load_llm(key):
    return ChatOpenAI(
        openai_api_key=key,
        temperature=0
    )

# -------------------
# Cached Vector DB
# -------------------

@st.cache_resource
def create_db(docs, key):
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    return FAISS.from_documents(docs, embeddings)

# -------------------
# Upload PDF
# -------------------

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file and api_key and st.session_state.db is None:

    with st.spinner("Analyzing document..."):

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        docs = splitter.split_documents(documents)

        db = create_db(docs, api_key)

        st.session_state.db = db

    st.success("Document ready! Ask questions.")

# -------------------
# Show Chat History
# -------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------
# Chat Input
# -------------------

question = st.chat_input("Ask something about the document")

if question and st.session_state.db and api_key:

    llm = load_llm(api_key)

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    results = st.session_state.db.similarity_search_with_score(question)

    relevant_docs = []

    for doc, score in results:
        if score < 0.7:
            relevant_docs.append(doc)

    if len(relevant_docs) == 0:

        response = "I could not find that in the document."

        with st.chat_message("assistant"):
            st.markdown(response)

    else:

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
You are a document assistant.

Answer the question ONLY using the provided context.

Rules:
- If the answer is not in the document, say: "I could not find that in the document."
- Do not invent information.
- Keep the answer clear and short.

Context:
{context}

Question:
{question}
"""

        response = llm.predict(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)

            st.markdown("**Sources:**")

            shown_pages = set()

            for doc in relevant_docs:
                page = doc.metadata.get("page", "?")

                if page not in shown_pages:
                    st.markdown(f"- Page {page}")
                    shown_pages.add(page)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
