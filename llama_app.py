from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import streamlit as st

st.title("Langchain-llama3.1 APP")

template = """Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="llama3.1")

chain = prompt | model

question = st.chat_input("Enter your question.")

if question:
    st.write(chain.invoke({"question": question}))