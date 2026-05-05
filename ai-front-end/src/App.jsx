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
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";

const transport = new DefaultChatTransport({
  api: "http://localhost:8000/api/chat",
});

function App() {
  const { messages, sendMessage, status } = useChat({ transport });

  const handleSubmit = (msg) => {
    if (!msg.text) return;
    sendMessage({ text: msg.text });
  };

  const renderPart = (part, idx) => {
    if (part.type === "text") {
      return <MessageResponse key={idx}>{part.text}</MessageResponse>;
    }
    if (part.type === "reasoning") {
      return (
        <MessageResponse key={idx} className="opacity-60 italic">
          {part.text}
        </MessageResponse>
      );
    }
    if (part.type?.startsWith("tool-") || part.type === "dynamic-tool") {
      const name = part.toolName ?? part.type.replace(/^tool-/, "");
      return (
        <pre
          key={idx}
          className="text-xs rounded bg-muted/50 px-2 py-1 overflow-x-auto"
        >
          {`🔧 ${name} (${part.state})${
            part.input ? `\nin:  ${JSON.stringify(part.input)}` : ""
          }${part.output ? `\nout: ${JSON.stringify(part.output)}` : ""}`}
        </pre>
      );
    }
    return null;
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
          {messages.map((message) => (
            <Message
              key={message.id}
              from={message.role === "user" ? "user" : "assistant"}
            >
              <MessageContent>
                {message.parts?.map(renderPart)}
              </MessageContent>
            </Message>
          ))}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      <footer className="flex-none px-6 py-4 border-t">
        <div className="mx-auto w-full max-w-2xl">
          <ChatInput onSubmit={handleSubmit} status={status} />
        </div>
      </footer>
    </div>
  );
}

export default App;
