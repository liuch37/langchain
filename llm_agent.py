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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun

#Write a date time tool to return current date and time.
def get_current_time(*args, **kwargs):
    """Return the current date and time in ISO format."""

    return datetime.now().isoformat()

# Write a web search tool
def get_query_from_web(query):
    """Return the search result from the websites."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

load_dotenv()

# build LLM
print("Building LLM......")
#llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)
llm = OllamaLLM(model="mistral")

# loading built-in tools
print("Loading tools......")
tools = load_tools(['llm-math'], llm=llm)
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
tools.extend(
    [datetime_tool,
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

question = "What is the birthday of Taiwan's current president? Then calculate his age in 2025."

# Creating an instance of StdOutCallbackHandler for managing standard output callbacks
handler = StdOutCallbackHandler()

# Processing a question through the initialized agent, utilizing the StdOutCallbackHandler for callback management
# The input is passed as a dictionary with "input" key containing the question
response = agent(
    {"input": question},
    callbacks=[handler]
)

print(response['output'])