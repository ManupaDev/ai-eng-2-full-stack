"""ModelMessage and content part types.

Mirrors `/tmp/ai-sdk-repo/packages/provider-utils/src/types/`:
- `model-message.ts` (the union)
- `system-model-message.ts`, `user-model-message.ts`, `assistant-model-message.ts`, `tool-model-message.ts`
- `content-part.ts` (TextPart, FilePart, ImagePart, ReasoningPart, ReasoningFilePart, CustomPart, ToolCallPart, ToolResultPart, ToolResultOutput)
- `tool-approval-request.ts`, `tool-approval-response.ts`

Plus `/tmp/ai-sdk-repo/packages/ai/src/prompt/`:
- `create-tool-model-output.ts`
- `message-conversion-error.ts`
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ProviderOptions = dict[str, Any]
ProviderReference = dict[str, Any]


class _Part(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


# ---------- Content parts ----------


class TextPart(_Part):
    type: Literal["text"] = "text"
    text: str
    provider_options: ProviderOptions | None = Field(
        default=None, alias="providerOptions"
    )


class ImagePart(_Part):
    """Deprecated; use FilePart with `media_type: 'image/...'`."""

    type: Literal["image"] = "image"
    image: Any  # DataContent | URL | ProviderReference
    media_type: str | None = Field(default=None, alias="mediaType")
    provider_options: ProviderOptions | None = Field(
        default=None, alias="providerOptions"
    )


class FilePart(_Part):
    type: Literal["file"] = "file"
    data: Any  # FileData | DataContent | URL | ProviderReference
    filename: str | None = None
    media_type: str = Field(alias="mediaType")
    provider_options: ProviderOptions | None = Field(
        default=None, alias="providerOptions"
    )


class ReasoningPart(_Part):
    type: Literal["reasoning"] = "reasoning"
    text: str
    provider_options: ProviderOptions | None = Field(
        default=None, alias="providerOptions"
    )


class ReasoningFilePart(_Part):
    type: Literal["reasoning-file"] = "reasoning-file"
    data: Any
    media_type: str = Field(alias="mediaType")
    provider_options: ProviderOptions | None = Field(
        default=None, alias="providerOptions"
    )


class CustomPart(_Part):
    type: Literal["custom"] = "custom"
    kind: str  # `{provider}.{provider-type}`
    provider_options: ProviderOptions | None = Field(
        default=None, alias="providerOptions"
    )


class ToolCallPart(_Part):
    type: Literal["tool-call"] = "tool-call"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    input: Any
    provider_options: ProviderOptions | None = Field(
        default=None, alias="providerOptions"
    )
    provider_executed: bool | None = Field(default=None, alias="providerExecuted")


class ToolResultPart(_Part):
    type: Literal["tool-result"] = "tool-result"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    output: Any  # ToolResultOutput
    provider_options: ProviderOptions | None = Field(
        default=None, alias="providerOptions"
    )


class ToolApprovalRequest(_Part):
    type: Literal["tool-approval-request"] = "tool-approval-request"
    approval_id: str = Field(alias="approvalId")
    tool_call_id: str = Field(alias="toolCallId")
    is_automatic: bool | None = Field(default=None, alias="isAutomatic")


class ToolApprovalResponse(_Part):
    type: Literal["tool-approval-response"] = "tool-approval-response"
    approval_id: str = Field(alias="approvalId")
    approved: bool
    reason: str | None = None
    provider_executed: bool | None = Field(default=None, alias="providerExecuted")


# ---------- ToolResultOutput variants ----------
# Modelled as plain dicts in the port — too many deprecated subtypes for strict typing.
# Producers/consumers should match the TS shape exactly.

ToolResultOutput = dict[str, Any]


# ---------- Model messages ----------


class SystemModelMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    role: Literal["system"] = "system"
    content: str
    provider_options: ProviderOptions | None = Field(
        default=None, alias="providerOptions"
    )


class UserModelMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    role: Literal["user"] = "user"
    content: Any  # str | list[TextPart | ImagePart | FilePart]
    provider_options: ProviderOptions | None = Field(
        default=None, alias="providerOptions"
    )


class AssistantModelMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    role: Literal["assistant"] = "assistant"
    content: Any  # str | list[TextPart | CustomPart | FilePart | ReasoningPart | ReasoningFilePart | ToolCallPart | ToolResultPart | ToolApprovalRequest]
    provider_options: ProviderOptions | None = Field(
        default=None, alias="providerOptions"
    )


class ToolModelMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    role: Literal["tool"] = "tool"
    content: Any  # list[ToolResultPart | ToolApprovalResponse]
    provider_options: ProviderOptions | None = Field(
        default=None, alias="providerOptions"
    )


ModelMessage = (
    SystemModelMessage | UserModelMessage | AssistantModelMessage | ToolModelMessage
)


# ---------- Errors ----------


class MessageConversionError(Exception):
    """ai/src/prompt/message-conversion-error.ts.

    Carries the original UIMessage that failed to convert.
    """

    name = "AI_MessageConversionError"

    def __init__(self, *, original_message: Any, message: str) -> None:
        super().__init__(message)
        self.original_message = original_message


# ---------- create_tool_model_output ----------


def _to_json_value(value: Any) -> Any:
    """Mirror TS `toJSONValue`: undefined → null."""
    return None if value is None else value


def _get_error_message(error: Any) -> str:
    """Mirror @ai-sdk/provider getErrorMessage."""
    if isinstance(error, BaseException):
        return str(error) or error.__class__.__name__
    if isinstance(error, str):
        return error
    try:
        return json.dumps(error)
    except Exception:
        return repr(error)


async def create_tool_model_output(
    *,
    tool_call_id: str,
    input: Any,
    output: Any,
    tool: Any | None,
    error_mode: Literal["none", "text", "json"],
) -> ToolResultOutput:
    """ai/src/prompt/create-tool-model-output.ts."""
    if error_mode == "text":
        return {"type": "error-text", "value": _get_error_message(output)}
    if error_mode == "json":
        return {"type": "error-json", "value": _to_json_value(output)}

    if tool is not None and getattr(tool, "to_model_output", None):
        return await tool.to_model_output(
            {"toolCallId": tool_call_id, "input": input, "output": output}
        )

    if isinstance(output, str):
        return {"type": "text", "value": output}
    return {"type": "json", "value": _to_json_value(output)}
