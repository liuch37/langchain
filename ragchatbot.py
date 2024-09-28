'''
This code is to build a Q&A chatbot using RAG.
'''

import streamlit as st
import time
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import pdb

load_dotenv()

st.title("RAG App building a QA chatbot given multiple PDF files")

print("Loading data......")
pdf_folder = './pdf'
print(os.listdir(pdf_folder))

# load multiple files in a folder
loaders = [PyPDFLoader(os.path.join(pdf_folder, fn)) for fn in os.listdir(pdf_folder)]

print(loaders)

alldocument = []
for loader in loaders:
    print("Loading raw document...", loader.file_path)
    raw_documents = loader.load()
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    doc = text_splitter.split_documents(raw_documents)
    alldocument.extend(doc)

client_settings = chromadb.Settings(chroma_server_host="localhost", chroma_server_http_port="8000")

vectorstore = Chroma.from_documents(documents=alldocument, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), client_settings=client_settings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 30})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("Ask me a question regarding your database!"):
    # Remove cache for chromadb
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    system_prompt = (
        "You are an virtual assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": query})
        st.write(response["answer"])
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
