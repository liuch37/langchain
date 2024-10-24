'''
This is a simple LLM agent calling external toolings.
'''

from dotenv import load_dotenv
import os
from datetime import datetime
import pdb

from langchain.callbacks import StdOutCallbackHandler
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama.llms import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# Setup database and LLM
print("Loading vector database......")
vectorstore = Chroma(embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), persist_directory='./chromadb', collection_metadata={"hnsw:space": "cosine"})
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 30})
print("Done loading vector database......")

# build LLM
print("Building LLM......")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_tokens=None, timeout=None)
#llm = OllamaLLM(model="mistral")

#Write a RAG tool to return the information of Chun-Hao Liu.
def get_chunhaol_data(query):
    """Return the information of Chun-Hao Liu from his database."""

    qa_system_prompt = (
        "You are an virtual assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})

    return response["answer"]

#Write a date time tool to return current date and time.
def get_current_time(*args, **kwargs):
    """Return the current date and time in ISO format."""

    return datetime.now().isoformat()

# Write a web search tool
def get_query_from_web(query):
    """Return the search result from the websites."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

# loading built-in tools
print("Loading tools......")
#tools = load_tools(['llm-math'], llm=llm)
# loading customized tools
datetime_tool = Tool(
        name="Datetime",
        func=get_current_time,
        description="Return the current date and time in ISO format",
    )
websearch_tool = Tool(
        name="WebSearch",
        func=get_query_from_web,
        description="Useful for web search",
    )
chunhao_tool = Tool(
        name="ChunHaoData",
        func=get_chunhaol_data,
        description="Return the information of Chun-Hao Liu from his database",
    )
tools = []
tools.extend(
    [chunhao_tool,
     websearch_tool]
    )

# Initializing an agent with specified parameters
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    return_intermediate_steps=True,
    verbose=True,
    handle_parsing_errors=True
)

question = "List all places that Chun-Hao may live til now. Use ChunHaoData tool."

# Creating an instance of StdOutCallbackHandler for managing standard output callbacks
handler = StdOutCallbackHandler()

# Processing a question through the initialized agent, utilizing the StdOutCallbackHandler for callback management
# The input is passed as a dictionary with "input" key containing the question
response = agent(
    {"input": question},
    callbacks=[handler]
)

print(response['output'])