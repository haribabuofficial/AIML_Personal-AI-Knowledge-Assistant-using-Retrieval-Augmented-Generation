import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

st.set_page_config(page_title="Multi-Doc RAG Chat", layout="wide")
st.title("📚 Multi-Document RAG Assistant")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    st.session_state.store = {}

# Sidebar - File Upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="You can upload multiple files at once"
    )

    if st.button("Process Documents", type="primary") and uploaded_files:
        with st.spinner("Processing your documents..."):
            documents = []
            
            for uploaded_file in uploaded_files:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name

                try:
                    if uploaded_file.name.endswith(".pdf"):
                        loader = PyPDFLoader(tmp_path)
                    elif uploaded_file.name.endswith(".txt"):
                        loader = TextLoader(tmp_path, encoding="utf-8")
                    elif uploaded_file.name.endswith(".docx"):
                        loader = UnstructuredWordDocumentLoader(tmp_path)
                    else:
                        st.warning(f"Unsupported file type: {uploaded_file.name}")
                        continue

                    documents.extend(loader.load())
                finally:
                    os.unlink(tmp_path)  # Clean up temp file

            if documents:
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                docs = text_splitter.split_documents(documents)

                # Create vector store
                embeddings = OpenAIEmbeddings()
                st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
                
                st.success(f"✅ Processed {len(uploaded_files)} document(s) successfully!")
            else:
                st.error("No valid documents processed.")

# Main Chat Interface
if st.session_state.vectorstore is None:
    st.info("👆 Please upload and process documents from the sidebar to start chatting.")
else:
    # Build RAG Chain (only once)
    if st.session_state.chain is None:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant.
            Answer ONLY based on the retrieved context below.
            If the answer is not found in the context, say "I don't know based on the provided documents."
            Do NOT use any outside knowledge.

            Context:
            {context}
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        llm = ChatOpenAI(temperature=0)

        def get_session_history(session_id):
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = InMemoryChatMessageHistory()
            return st.session_state.store[session_id]

        rag_chain = (
            {
                "context": RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
                "question": RunnableLambda(lambda x: x["question"]),
                "history": RunnableLambda(lambda x: x["history"]),
            }
            | prompt
            | llm
        )

        st.session_state.chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain.invoke(
                    {"question": prompt},
                    config={"configurable": {"session_id": "user1"}}
                )
                st.markdown(response.content)
        
        st.session_state.messages.append({"role": "assistant", "content": response.content})