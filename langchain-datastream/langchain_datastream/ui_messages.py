"""UIMessage and UIMessagePart variants.

Mirrors `/tmp/ai-sdk-repo/packages/ai/src/ui/ui-messages.ts`.

Key points:
- Each part has a discriminator on `type`. We use Pydantic v2 with `populate_by_name`
  and aliases for camelCase fields (e.g. `provider_metadata` ↔ `providerMetadata`).
- Tool parts use a flexible state-keyed shape because TS uses an internal union
  with state-specific field constraints; we accept any combo and validate at
  conversion time. See `state` field for the seven valid states.
- `data-{name}` parts are matched by prefix.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ProviderMetadata = dict[str, Any]
ProviderReference = dict[str, Any]


class _Part(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


# ---------- Part variants ----------


class TextUIPart(_Part):
    type: Literal["text"] = "text"
    text: str
    state: Literal["streaming", "done"] | None = None
    provider_metadata: ProviderMetadata | None = Field(
        default=None, alias="providerMetadata"
    )


class CustomContentUIPart(_Part):
    type: Literal["custom"] = "custom"
    kind: str  # `{provider}.{provider-type}`
    provider_metadata: ProviderMetadata | None = Field(
        default=None, alias="providerMetadata"
    )


class ReasoningUIPart(_Part):
    type: Literal["reasoning"] = "reasoning"
    text: str
    state: Literal["streaming", "done"] | None = None
    provider_metadata: ProviderMetadata | None = Field(
        default=None, alias="providerMetadata"
    )


class SourceUrlUIPart(_Part):
    type: Literal["source-url"] = "source-url"
    source_id: str = Field(alias="sourceId")
    url: str
    title: str | None = None
    provider_metadata: ProviderMetadata | None = Field(
        default=None, alias="providerMetadata"
    )


class SourceDocumentUIPart(_Part):
    type: Literal["source-document"] = "source-document"
    source_id: str = Field(alias="sourceId")
    media_type: str = Field(alias="mediaType")
    title: str
    filename: str | None = None
    provider_metadata: ProviderMetadata | None = Field(
        default=None, alias="providerMetadata"
    )


class FileUIPart(_Part):
    type: Literal["file"] = "file"
    media_type: str = Field(alias="mediaType")
    filename: str | None = None
    url: str
    provider_reference: ProviderReference | None = Field(
        default=None, alias="providerReference"
    )
    provider_metadata: ProviderMetadata | None = Field(
        default=None, alias="providerMetadata"
    )


class ReasoningFileUIPart(_Part):
    type: Literal["reasoning-file"] = "reasoning-file"
    media_type: str = Field(alias="mediaType")
    url: str
    provider_metadata: ProviderMetadata | None = Field(
        default=None, alias="providerMetadata"
    )


class StepStartUIPart(_Part):
    type: Literal["step-start"] = "step-start"


class DataUIPart(_Part):
    """`type: 'data-<name>'` with arbitrary payload."""

    type: str  # 'data-...'
    id: str | None = None
    data: Any

    @field_validator("type")
    @classmethod
    def _check_data_prefix(cls, v: str) -> str:
        if not v.startswith("data-"):
            raise ValueError(f"DataUIPart type must start with 'data-', got {v!r}")
        return v


# ---------- Tool parts ----------

ToolState = Literal[
    "input-streaming",
    "input-available",
    "approval-requested",
    "approval-responded",
    "output-available",
    "output-error",
    "output-denied",
]


class ToolApproval(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    id: str
    approved: bool | None = None
    reason: str | None = None
    is_automatic: bool | None = Field(default=None, alias="isAutomatic")


class _ToolPartBase(_Part):
    """Shared fields for `ToolUIPart` and `DynamicToolUIPart`."""

    tool_call_id: str = Field(alias="toolCallId")
    title: str | None = None
    provider_executed: bool | None = Field(default=None, alias="providerExecuted")
    state: ToolState
    input: Any | None = None
    raw_input: Any | None = Field(default=None, alias="rawInput")
    output: Any | None = None
    error_text: str | None = Field(default=None, alias="errorText")
    call_provider_metadata: ProviderMetadata | None = Field(
        default=None, alias="callProviderMetadata"
    )
    result_provider_metadata: ProviderMetadata | None = Field(
        default=None, alias="resultProviderMetadata"
    )
    preliminary: bool | None = None
    approval: ToolApproval | None = None


class ToolUIPart(_ToolPartBase):
    """`type: 'tool-<name>'`."""

    type: str  # 'tool-...'

    @field_validator("type")
    @classmethod
    def _check_tool_prefix(cls, v: str) -> str:
        if not v.startswith("tool-"):
            raise ValueError(f"ToolUIPart type must start with 'tool-', got {v!r}")
        if v == "tool-":
            raise ValueError("ToolUIPart type must include a name after 'tool-'")
        return v


class DynamicToolUIPart(_ToolPartBase):
    type: Literal["dynamic-tool"] = "dynamic-tool"
    tool_name: str = Field(alias="toolName")


UIMessagePart = (
    TextUIPart
    | CustomContentUIPart
    | ReasoningUIPart
    | SourceUrlUIPart
    | SourceDocumentUIPart
    | FileUIPart
    | ReasoningFileUIPart
    | StepStartUIPart
    | DataUIPart
    | ToolUIPart
    | DynamicToolUIPart
)


# ---------- UIMessage ----------


class UIMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    id: str | None = None
    role: Literal["system", "user", "assistant"]
    metadata: Any | None = None
    parts: list[Any] = Field(default_factory=list)
    """Parts are validated lazily — use ``validate_part`` to coerce a raw dict."""


# ---------- Type guards (runtime, mirroring ui-messages.ts) ----------


def is_text_ui_part(part: Any) -> bool:
    return _part_type(part) == "text"


def is_custom_content_ui_part(part: Any) -> bool:
    return _part_type(part) == "custom"


def is_file_ui_part(part: Any) -> bool:
    return _part_type(part) == "file"


def is_reasoning_file_ui_part(part: Any) -> bool:
    return _part_type(part) == "reasoning-file"


def is_reasoning_ui_part(part: Any) -> bool:
    return _part_type(part) == "reasoning"


def is_source_url_ui_part(part: Any) -> bool:
    return _part_type(part) == "source-url"


def is_source_document_ui_part(part: Any) -> bool:
    return _part_type(part) == "source-document"


def is_step_start_ui_part(part: Any) -> bool:
    return _part_type(part) == "step-start"


def is_data_ui_part(part: Any) -> bool:
    t = _part_type(part)
    return isinstance(t, str) and t.startswith("data-")


def is_static_tool_ui_part(part: Any) -> bool:
    t = _part_type(part)
    return isinstance(t, str) and t.startswith("tool-")


def is_dynamic_tool_ui_part(part: Any) -> bool:
    return _part_type(part) == "dynamic-tool"


def is_tool_ui_part(part: Any) -> bool:
    return is_static_tool_ui_part(part) or is_dynamic_tool_ui_part(part)


def get_static_tool_name(part: Any) -> str:
    """`tool-foo-bar` → `foo-bar`."""
    t = _part_type(part)
    if not isinstance(t, str) or not t.startswith("tool-"):
        raise ValueError(f"Not a static tool part: {part!r}")
    return t[len("tool-") :]


def get_tool_name(part: Any) -> str:
    """ui-messages.ts:585 — name for static or dynamic tool parts."""
    if is_dynamic_tool_ui_part(part):
        return _part_field(part, "tool_name", "toolName")
    return get_static_tool_name(part)


# ---------- Helpers ----------


def _part_type(part: Any) -> Any:
    if isinstance(part, dict):
        return part.get("type")
    return getattr(part, "type", None)


def _part_field(part: Any, py_name: str, alias: str) -> Any:
    if isinstance(part, dict):
        return part.get(alias) if alias in part else part.get(py_name)
    return getattr(part, py_name, getattr(part, alias, None))


_PART_MODELS: dict[str, type[_Part]] = {
    "text": TextUIPart,
    "custom": CustomContentUIPart,
    "reasoning": ReasoningUIPart,
    "source-url": SourceUrlUIPart,
    "source-document": SourceDocumentUIPart,
    "file": FileUIPart,
    "reasoning-file": ReasoningFileUIPart,
    "step-start": StepStartUIPart,
    "dynamic-tool": DynamicToolUIPart,
}


def validate_part(raw: dict[str, Any]) -> _Part:
    """Coerce a raw part dict into the appropriate Pydantic model."""
    t = raw.get("type")
    if not isinstance(t, str):
        raise ValueError(f"Part missing string `type`: {raw!r}")
    if t in _PART_MODELS:
        return _PART_MODELS[t].model_validate(raw)
    if t.startswith("tool-"):
        return ToolUIPart.model_validate(raw)
    if t.startswith("data-"):
        return DataUIPart.model_validate(raw)
    raise ValueError(f"Unknown part type: {t!r}")
