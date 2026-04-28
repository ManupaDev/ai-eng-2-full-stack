import ChatInput from "@/components/ChatInput";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import { getMessages, sendMessage } from "@/lib/api";
import { useEffect, useState } from "react";

function App() {
  const [messages, setMessages] = useState([]);

  const fetchMessages = async () => {
    const msgs = await getMessages();
    setMessages(msgs);
  };

  useEffect(() => {
    fetchMessages();
  }, []);

  const handleSubmit = async (message) => {
   const newMessages = await sendMessage(message.text);
   setMessages(newMessages);
  };

  return (
    <div className="flex flex-col h-screen">
      <header className="flex-none px-6 py-4 border-b">
        <div className="mx-auto w-full max-w-2xl">
          <h1 className="text-2xl font-bold">Welcome to the AI Frontend</h1>
        </div>
      </header>

      <Conversation className="flex-1">
        <ConversationContent className="mx-auto w-full max-w-2xl">
          {messages.map((message, i) => (
            <Message key={i} from={message.role === "user" ? "user" : "assistant"}>
              <MessageContent>
                <MessageResponse>{message.content}</MessageResponse>
              </MessageContent>
            </Message>
          ))}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      <footer className="flex-none px-6 py-4 border-t">
        <div className="mx-auto w-full max-w-2xl">
          <ChatInput onSubmit={handleSubmit} />
        </div>
      </footer>
    </div>
  );
}

export default App;
