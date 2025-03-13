import os
from getpass import getpass
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
from pydantic import BaseModel, Field  # Updated import
from langchain.agents import AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere.chat_models import ChatCohere
from langchain_core.messages import HumanMessage, AIMessage
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set environment variables
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# **Define Tools**

# Web Search Tool
internet_search = TavilySearchResults()
internet_search.name = "internet_search"
internet_search.description = "Returns a list of relevant document snippets for a textual query retrieved from the internet."
class TavilySearchInput(BaseModel):
    query: str = Field(description="Query to search the internet with")
internet_search.args_schema = TavilySearchInput

# Python Interpreter Tool
python_repl = PythonREPL()
python_tool = Tool(
    name="python_interpreter",
    description="Executes Python code and returns the result. Runs in a static sandbox without interactive mode; use print() for output.",
    func=python_repl.run,
)
class ToolInput(BaseModel):
    code: str = Field(description="Python code to execute")
python_tool.args_schema = ToolInput

# Custom Random Operation Tool
@tool
def random_operation_tool(a: int, b: int) -> str:
    """Calculates a random operation (addition or multiplication) between two numbers."""
    coin_toss = random.uniform(0, 1)
    if coin_toss > 0.5:
        result = a * b
        operation = "multiplication"
    else:
        result = a + b
        operation = "addition"
    return f"The result of {operation} between {a} and {b} is {result}"
random_operation_tool.name = "random_operation"
random_operation_tool.description = "Calculates a random operation between two numbers."
class RandomOperationInputs(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")
random_operation_tool.args_schema = RandomOperationInputs

# **Set Up the Agent**

# Initialize the LLM
llm = ChatCohere(model="command-r-plus", temperature=0.3)

# Define the preamble
preamble = """
You are an expert assistant who answers user queries using the most relevant tools. You are equipped with:
- 'internet_search': for web searches
- 'python_interpreter': for executing Python code
- 'random_operation': for computing a random operation between two numbers (use this when asked to do so)
If no tool is needed, respond directly without invoking a tool.
"""

# List of tools
tools = [internet_search, python_tool, random_operation_tool]

# Initialize chat history
chat_history = []

# **Terminal UI/UX**

# Welcome message
print("\n" + "="*50)
print("Welcome to the Cohere AI Chat Agent!")
print("I'm here to assist with internet searches, Python code execution, and random number operations.")
print("Type your query below or enter 'quit' to exit.")
print("="*50 + "\n")

# Main chat loop
while True:
    user_input = input("User: ")
    if user_input.lower() in ['quit', 'exit']:
        print("\nGoodbye! Thanks for chatting with me.")
        break
    
    # Add user input to chat history
    chat_history.append(HumanMessage(content=user_input))
    
    # Create prompt with chat history
    prompt = ChatPromptTemplate.from_messages(chat_history)
    
    # Create the ReAct agent
    agent = create_cohere_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    
    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Invoke the agent
    response = agent_executor.invoke({"preamble": preamble})
    
    # Display the agent's response
    print("\nAgent:", response['output'])
    
    # Add agent response to chat history
    chat_history.append(AIMessage(content=response['output']))