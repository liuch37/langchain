'''
How to run this app:
1. Install necessary libraries.
2. Download ollama (https://ollama.com/) and install its app on your computer.
3. Download llama3.1 model by running "ollama pull llama3.1".
4. Run "streamlit run llama_app.py"
'''

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import streamlit as st

st.title("Langchain-llama3.1 APP")

template = """Context: {context}
Response:"""

prompting = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="llama3.1")

chain = prompting | model

#question = st.chat_input("Enter your question.")

#if question:
#    st.write(chain.invoke({"question": question}))

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How is it going?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        messages = ""
        for m in st.session_state.messages:
            messages += m["role"]+":"+m["content"]+"\n"
        response = chain.invoke({"context": messages})
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})