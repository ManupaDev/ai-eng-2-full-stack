"""UIMessageChunk variants emitted by the data stream protocol.

Each chunk is serialized to JSON and framed as an SSE `data:` line.
Field names match the TypeScript spec exactly (camelCase) — Pydantic
aliases handle the Python snake_case ↔ JSON camelCase boundary.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Chunk(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class StartChunk(_Chunk):
    type: Literal["start"] = "start"
    message_id: str | None = Field(default=None, alias="messageId")


class StartStepChunk(_Chunk):
    type: Literal["start-step"] = "start-step"


class FinishStepChunk(_Chunk):
    type: Literal["finish-step"] = "finish-step"


class FinishChunk(_Chunk):
    type: Literal["finish"] = "finish"


class AbortChunk(_Chunk):
    type: Literal["abort"] = "abort"
    reason: str | None = None


class ErrorChunk(_Chunk):
    type: Literal["error"] = "error"
    error_text: str = Field(alias="errorText")


class TextStartChunk(_Chunk):
    type: Literal["text-start"] = "text-start"
    id: str


class TextDeltaChunk(_Chunk):
    type: Literal["text-delta"] = "text-delta"
    id: str
    delta: str


class TextEndChunk(_Chunk):
    type: Literal["text-end"] = "text-end"
    id: str


class ReasoningStartChunk(_Chunk):
    type: Literal["reasoning-start"] = "reasoning-start"
    id: str


class ReasoningDeltaChunk(_Chunk):
    type: Literal["reasoning-delta"] = "reasoning-delta"
    id: str
    delta: str


class ReasoningEndChunk(_Chunk):
    type: Literal["reasoning-end"] = "reasoning-end"
    id: str


class ToolInputStartChunk(_Chunk):
    type: Literal["tool-input-start"] = "tool-input-start"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    dynamic: bool | None = None


class ToolInputDeltaChunk(_Chunk):
    type: Literal["tool-input-delta"] = "tool-input-delta"
    tool_call_id: str = Field(alias="toolCallId")
    input_text_delta: str = Field(alias="inputTextDelta")


class ToolInputAvailableChunk(_Chunk):
    type: Literal["tool-input-available"] = "tool-input-available"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    input: Any
    dynamic: bool | None = None


class ToolOutputAvailableChunk(_Chunk):
    type: Literal["tool-output-available"] = "tool-output-available"
    tool_call_id: str = Field(alias="toolCallId")
    output: Any


class ToolOutputErrorChunk(_Chunk):
    type: Literal["tool-output-error"] = "tool-output-error"
    tool_call_id: str = Field(alias="toolCallId")
    error_text: str = Field(alias="errorText")


class ToolApprovalRequestChunk(_Chunk):
    type: Literal["tool-approval-request"] = "tool-approval-request"
    approval_id: str = Field(alias="approvalId")
    tool_call_id: str = Field(alias="toolCallId")


class FileChunk(_Chunk):
    type: Literal["file"] = "file"
    media_type: str = Field(alias="mediaType")
    url: str


class SourceUrlChunk(_Chunk):
    type: Literal["source-url"] = "source-url"
    source_id: str = Field(alias="sourceId")
    url: str
    title: str | None = None


class SourceDocumentChunk(_Chunk):
    type: Literal["source-document"] = "source-document"
    source_id: str = Field(alias="sourceId")
    media_type: str = Field(alias="mediaType")
    title: str
    filename: str | None = None


class DataChunk(_Chunk):
    """Custom data part. The `type` is `data-<name>` chosen by the producer.

    The Pydantic model accepts any string starting with `data-`.
    """

    type: str
    data: Any
    id: str | None = None
    transient: bool | None = None


UIMessageChunk = (
    StartChunk
    | StartStepChunk
    | FinishStepChunk
    | FinishChunk
    | AbortChunk
    | ErrorChunk
    | TextStartChunk
    | TextDeltaChunk
    | TextEndChunk
    | ReasoningStartChunk
    | ReasoningDeltaChunk
    | ReasoningEndChunk
    | ToolInputStartChunk
    | ToolInputDeltaChunk
    | ToolInputAvailableChunk
    | ToolOutputAvailableChunk
    | ToolOutputErrorChunk
    | ToolApprovalRequestChunk
    | FileChunk
    | SourceUrlChunk
    | SourceDocumentChunk
    | DataChunk
)


def chunk_to_dict(chunk: UIMessageChunk) -> dict[str, Any]:
    """Serialize a chunk to its JSON dict shape with camelCase keys and no None values."""
    return chunk.model_dump(by_alias=True, exclude_none=True)
