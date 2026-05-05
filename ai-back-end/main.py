from dotenv import load_dotenv

load_dotenv(".env.local")

import os
from typing import Annotated, Any, Sequence, TypedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_classic.agents import load_tools
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from ai_sdk import to_base_messages, ui_message_stream_response

# --- Agent setup ---


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


llm = ChatOpenAI(model="gpt-4o")

tools: list[Any] = []
if os.environ.get("TAVILY_API_KEY"):
    tools.append(TavilySearch())
if os.environ.get("OPENWEATHERMAP_API_KEY"):
    tools.append(load_tools(["openweathermap-api"], llm)[0])

llm_with_tools = llm.bind_tools(tools) if tools else llm


def llm_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are an intelligent AI assistant.")
    response = llm_with_tools.invoke([system_prompt] + list(state["messages"]))
    return {"messages": [response]}


def decision(state: AgentState):
    last = state["messages"][-1]
    return "continue" if getattr(last, "tool_calls", None) else "end"


graph = StateGraph(AgentState)
graph.add_node("agent", llm_call)
if tools:
    graph.add_node("tools", ToolNode(tools=tools))
graph.set_entry_point("agent")
if tools:
    graph.add_conditional_edges("agent", decision, {"continue": "tools", "end": END})
    graph.add_edge("tools", "agent")
else:
    graph.add_edge("agent", END)
agent = graph.compile()

# --- FastAPI ---

server = FastAPI()

server.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    messages: list[Any]
    """UIMessage[] from the frontend `useChat` hook."""


@server.post("/api/chat")
async def chat(req: ChatRequest):
    """Streams the agent's response in AI SDK v5 Data Stream Protocol format."""
    base_messages = await to_base_messages(req.messages)
    stream = agent.astream(
        {"messages": base_messages}, stream_mode=["values", "messages"]
    )
    return ui_message_stream_response(stream)
