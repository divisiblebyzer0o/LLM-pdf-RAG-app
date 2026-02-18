import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="PDF Chat", layout="wide")
st.title("📄 Chat With Your PDF (Offline RAG)")

# ---------- LOAD COMPONENTS (RUNS ONCE) ----------
@st.cache_resource
def load_rag():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k":8}
    )

    llm = Ollama(
        model="phi3",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
Use ONLY the context below.
If answer not present say: I don't know.

Context:
{context}

Question:
{question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain , retriever





rag_chain , retriever = load_rag()


# ---------- CHAT MEMORY ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------- USER INPUT ----------
if prompt := st.chat_input("Ask something about the PDF..."):

    # show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    docs = retriever.invoke(prompt)
    st.write("DEBUG docs:", docs)

    # generate response
    with st.chat_message("assistant"):
 
        response = rag_chain.invoke(prompt)
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )


