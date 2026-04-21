import {
  PromptInput,
  PromptInputBody,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
} from "@/components/ai-elements/prompt-input";

const ChatInput = ({ onSubmit, status = "ready" }) => {
  const handleSubmit = (message) => {
    if (!message.text) return;
    onSubmit?.(message);
  };

  return (
    <PromptInput onSubmit={handleSubmit}>
      <PromptInputBody>
        <PromptInputTextarea />
      </PromptInputBody>
      <PromptInputFooter>
        <PromptInputSubmit status={status} />
      </PromptInputFooter>
    </PromptInput>
  );
};

export default ChatInput;
