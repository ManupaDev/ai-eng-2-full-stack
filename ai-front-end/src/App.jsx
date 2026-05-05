import ChatInput from "@/components/ChatInput";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from "@/components/ai-elements/reasoning";
import { Suggestion, Suggestions } from "@/components/ai-elements/suggestion";
import {
  Tool,
  ToolContent,
  ToolHeader,
  ToolInput,
  ToolOutput,
} from "@/components/ai-elements/tool";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { MessageSquareIcon } from "lucide-react";

const transport = new DefaultChatTransport({
  api: "http://localhost:8000/api/chat",
});

const SUGGESTIONS = [
  "What's the temperature in Tokyo right now?",
  "Search the news for AI breakthroughs this week",
  "Tell me a fun fact about octopuses",
];

function App() {
  const { messages, sendMessage, status } = useChat({ transport });

  const submit = (text) => {
    if (!text?.trim()) return;
    sendMessage({ text });
  };

  return (
    <div className="flex flex-col h-screen">
      <header className="flex-none border-b">
        <div className="mx-auto w-full max-w-3xl px-6 py-4">
          <h1 className="text-xl font-semibold">AI Assistant</h1>
          <p className="text-xs text-muted-foreground">
            Streaming via Vercel AI SDK · LangGraph backend · live tools
          </p>
        </div>
      </header>

      <Conversation className="flex-1">
        <ConversationContent className="mx-auto w-full max-w-3xl">
          {messages.length === 0 ? (
            <ConversationEmptyState
              icon={<MessageSquareIcon className="size-10" />}
              title="Start a conversation"
              description="Ask anything — the agent has live web search and weather tools."
            />
          ) : (
            messages.map((message) => (
              <Message key={message.id} from={message.role}>
                <MessageContent>
                  {message.parts?.map((part, idx) =>
                    renderPart(part, `${message.id}-${idx}`, status)
                  )}
                </MessageContent>
              </Message>
            ))
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      {messages.length === 0 && (
        <div className="flex-none border-t bg-muted/30">
          <div className="mx-auto w-full max-w-3xl px-6 py-3">
            <p className="mb-2 text-xs font-medium text-muted-foreground">
              Try asking
            </p>
            <Suggestions>
              {SUGGESTIONS.map((s) => (
                <Suggestion key={s} suggestion={s} onClick={submit} />
              ))}
            </Suggestions>
          </div>
        </div>
      )}

      <footer className="flex-none border-t">
        <div className="mx-auto w-full max-w-3xl px-6 py-4">
          <ChatInput
            onSubmit={(msg) => submit(msg.text)}
            status={status}
          />
        </div>
      </footer>
    </div>
  );
}

function renderPart(part, key, status) {
  if (!part?.type) return null;

  if (part.type === "text") {
    return <MessageResponse key={key}>{part.text}</MessageResponse>;
  }

  if (part.type === "reasoning") {
    const isStreaming = status === "streaming";
    return (
      <Reasoning key={key} isStreaming={isStreaming} className="mb-3">
        <ReasoningTrigger />
        <ReasoningContent>{part.text}</ReasoningContent>
      </Reasoning>
    );
  }

  if (part.type === "step-start") {
    // Not visually rendered — the Tool blocks make the steps obvious enough.
    return null;
  }

  if (part.type === "dynamic-tool" || part.type?.startsWith("tool-")) {
    return (
      <Tool key={key} defaultOpen={part.state !== "output-available"}>
        <ToolHeader
          type={part.type}
          state={part.state}
          toolName={part.toolName}
        />
        <ToolContent>
          {part.input != null && <ToolInput input={part.input} />}
          <ToolOutput
            output={
              part.output != null ? (
                <pre className="whitespace-pre-wrap text-xs">
                  {typeof part.output === "string"
                    ? part.output
                    : JSON.stringify(part.output, null, 2)}
                </pre>
              ) : null
            }
            errorText={part.errorText}
          />
        </ToolContent>
      </Tool>
    );
  }

  return null;
}

export default App;
