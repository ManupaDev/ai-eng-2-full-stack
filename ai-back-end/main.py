from dotenv import load_dotenv

load_dotenv(".env.local")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

server = FastAPI()

server.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

messages = []

model = ChatOpenAI(model="gpt-4o")
chain = model | StrOutputParser()


class Message(BaseModel):
    text: str


@server.get("/api/messages")
def get_messages():
    return {"messages": messages}


@server.post("/api/messages")
def post_message(message: Message):
    messages.append({"role": "user", "content": message.text})
    reply = chain.invoke(messages)
    messages.append({"role": "assistant", "content": reply})
    return {"messages": messages}
