from utils import chunk_document, chunk_text_by_size, chunk_by_sentences, chunk_by_words, chunk_by_paragraphs

import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.tools import Tool

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-70b-8192")

def chunker_tool(query: str) -> str:
    """
    Tool to chunk text based on different methods.
    """
    try:
        chunks = chunk_by_sentences(query)
        return f"Chunked text: {chunks}"
    except Exception as e:
        return f"Error chunking text: {str(e)}"
    
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
    template="""You are an intelligent agent that uses tools to answer questions.
    Use the following tools to help you answer the question:
    {tools}
    Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
{agent_scratchpad}
"""
)

all_tools = load_tools(llm = llm) + [chunker_tool]

agent_with_all_tools = create_react_agent(
    tools=all_tools,
    llm=llm,
    prompt=prompt,
)

AgentExecutor(
    agent=agent_with_all_tools,
    tools=all_tools,
    verbose=True
)

if __name__ == "__main__":
    result = AgentExecutor.invoke({"input": input("Enter your question: ")})
    print(f"Result: {result}")