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

vectorstore = Chroma.from_documents(documents=alldocument, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

query = st.chat_input("Ask me a question!")

prompt = query

system_prompt = (
    "You are an assistant for question-answering tasks. "
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

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})

    st.write(response["answer"])
