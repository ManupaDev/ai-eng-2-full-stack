from dotenv import load_dotenv

load_dotenv(".env.local")

from typing import Annotated, Sequence, TypedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_classic.agents import load_tools
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

# --- Agent setup ---

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


llm = ChatOpenAI(model="gpt-4o")

search_tool = TavilySearch()
weather_tool = load_tools(["openweathermap-api"], llm)[0]
tools = [search_tool, weather_tool]

llm_with_tools = llm.bind_tools(tools)


def llm_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are an intelligent AI assistant.")
    response = llm_with_tools.invoke([system_prompt] + list(state["messages"]))
    return {"messages": [response]}


def decision(state: AgentState):
    last = state["messages"][-1]
    return "continue" if last.tool_calls else "end"


graph = StateGraph(AgentState)
graph.add_node("agent", llm_call)
graph.add_node("tools", ToolNode(tools=tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", decision, {"continue": "tools", "end": END})
graph.add_edge("tools", "agent")
agent = graph.compile()

# --- FastAPI ---

server = FastAPI()

server.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

messages = []


class Message(BaseModel):
    text: str


def to_lc_messages(msgs):
    result = []
    for m in msgs:
        if m["role"] == "user":
            result.append(HumanMessage(content=m["content"]))
        else:
            result.append(AIMessage(content=m["content"]))
    return result


@server.get("/api/messages")
def get_messages():
    return {"messages": messages}


@server.post("/api/messages")
def post_message(message: Message):
    messages.append({"role": "user", "content": message.text})
    result = agent.invoke({"messages": to_lc_messages(messages)})
    reply = result["messages"][-1].content
    messages.append({"role": "assistant", "content": reply})
    return {"messages": messages}
