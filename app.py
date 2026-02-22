import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="PDF Chat", layout="wide")
st.title("📄 Chat With Your PDF (Offline RAG)")

upload_files = st.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True)

def ingest_uploaded_pdfs(files , vectorstore, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)

    new_docs = []

    for files in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(files.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        for d in documents:
            d.metadata["source_file"] = files.name
            new_docs.append(d)

    chunks = splitter.split_documents(new_docs)
    vectorstore.add_documents(chunks)
    return vectorstore 


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
        search_kwargs={"k":8 , "fetch_k":15}
    )

    llm = Ollama(
        model="phi3",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
You are a grounded QA assistant.

Use context ONLY below the answer.
Answer mention page numbers when possible.
Do not repeat question.
                                              Do not include the word "Question" in your answer.
Use all the relevant information from the context to answer the question.
                                              Return only the final answer without any additional commentary.

Context:
{context}

Question:
{question}
""")

    def format_docs(docs):
        formatted = []
        for doc in docs:
            page = doc.metadata.get("page_label", doc.metadata.get("page"))
            text = f"(page{page}) {doc.page_content}"
            formatted.append(text)

        return "\n\n".join(formatted)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain , retriever , vectorstore , embeddings





rag_chain , retriever, vectorstore, embeddings = load_rag()

if upload_files:
    with st.spinner("Ingesting PDFs..."):
        vectorstore = ingest_uploaded_pdfs(upload_files , vectorstore, embeddings)
      

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k":5 , "fetch_k":10} )

        st.success("PDFs ingested successfully!")  


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
    
    with st.expander(f"🔍 Retrieved {len(docs)} chunks from the PDF"):
        for doc in docs:
            st.write(f"Page {doc.metadata.get('page_label', doc.metadata.get('page'))}")
            st.write(doc.page_content)
                     

    # generate response
    with st.chat_message("assistant"):

        message_placeholder = st.empty()
        full_response = ""

        for chunk in rag_chain.stream(prompt):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response) 
 
        

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )


