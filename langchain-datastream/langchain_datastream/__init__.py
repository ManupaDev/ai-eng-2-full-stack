"""Python port of the Vercel AI SDK LangChain adapter.

Mirrors `@ai-sdk/langchain` v1.x. Public surface is intentionally small;
internals match the TS layout for ease of cross-reference.
"""

from .adapter import to_ui_message_stream
from .callbacks import StreamCallbacks
from .chunks import (
    AbortChunk,
    DataChunk,
    ErrorChunk,
    FileChunk,
    FinishChunk,
    FinishStepChunk,
    ReasoningDeltaChunk,
    ReasoningEndChunk,
    ReasoningStartChunk,
    SourceDocumentChunk,
    SourceUrlChunk,
    StartChunk,
    StartStepChunk,
    TextDeltaChunk,
    TextEndChunk,
    TextStartChunk,
    ToolApprovalRequestChunk,
    ToolInputAvailableChunk,
    ToolInputDeltaChunk,
    ToolInputStartChunk,
    ToolOutputAvailableChunk,
    ToolOutputErrorChunk,
    UIMessageChunk,
    chunk_to_dict,
)
from .convert_to_model_messages import convert_to_model_messages

try:
    # FastAPI is an optional integration — only loaded when fastapi is installed.
    from .fastapi import DATA_STREAM_HEADERS, ui_message_stream_response
except ModuleNotFoundError:  # pragma: no cover - exercised when fastapi missing
    DATA_STREAM_HEADERS = None  # type: ignore[assignment]
    ui_message_stream_response = None  # type: ignore[assignment]

from .input import (
    convert_assistant_content,
    convert_model_messages,
    convert_tool_result_part,
    convert_user_content,
    to_base_messages,
)
from .model_messages import MessageConversionError, create_tool_model_output
from .sse import encode_sse_chunk, encode_sse_done, frame_data_stream
from .types import (
    GPT5ReasoningOutput,
    ImageGenerationOutput,
    LangGraphEventState,
    ModelStreamState,
    ReasoningContentBlock,
    ThinkingContentBlock,
)
from .ui_messages import UIMessage

__all__ = [
    "AbortChunk",
    "DATA_STREAM_HEADERS",
    "DataChunk",
    "ErrorChunk",
    "FileChunk",
    "FinishChunk",
    "FinishStepChunk",
    "GPT5ReasoningOutput",
    "ImageGenerationOutput",
    "LangGraphEventState",
    "MessageConversionError",
    "ModelStreamState",
    "ReasoningContentBlock",
    "ReasoningDeltaChunk",
    "ReasoningEndChunk",
    "ReasoningStartChunk",
    "SourceDocumentChunk",
    "SourceUrlChunk",
    "StartChunk",
    "StartStepChunk",
    "StreamCallbacks",
    "TextDeltaChunk",
    "TextEndChunk",
    "TextStartChunk",
    "ThinkingContentBlock",
    "ToolApprovalRequestChunk",
    "ToolInputAvailableChunk",
    "ToolInputDeltaChunk",
    "ToolInputStartChunk",
    "ToolOutputAvailableChunk",
    "ToolOutputErrorChunk",
    "UIMessage",
    "UIMessageChunk",
    "chunk_to_dict",
    "convert_assistant_content",
    "convert_model_messages",
    "convert_to_model_messages",
    "convert_tool_result_part",
    "convert_user_content",
    "create_tool_model_output",
    "encode_sse_chunk",
    "encode_sse_done",
    "frame_data_stream",
    "to_base_messages",
    "to_ui_message_stream",
    "ui_message_stream_response",
]
