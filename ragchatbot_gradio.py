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
import gradio as gr
import pdb

load_dotenv()

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

# Setup database and LLM
vectorstore = Chroma.from_documents(documents=alldocument, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), persist_directory='./chromadb')
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 30})
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

# Accept user input
def ask(query, history):
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
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})

    return response["answer"]

# lanuch gradio
gr.ChatInterface(
    ask,
    chatbot=gr.Chatbot(height=400),
    textbox=gr.Textbox(placeholder="Ask me a question regarding Chun-Hao Liu.", container=False, scale=7),
    title="Chatbot",
    description="Ask Chatbot any academia/resume question regarding Chun-Hao Liu",
    theme="soft",
    examples=["Tell me about Chun-Hao", "List three papers authored by Chun-Hao", "Where does Chun-Hao live?"],
    cache_examples=True,    
).launch(share=True)