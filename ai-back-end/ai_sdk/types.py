"""Shared types for the LangChain adapter port.

Mirrors `/tmp/ai-sdk-repo/packages/langchain/src/types.ts`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class ReasoningContentBlock(TypedDict):
    type: str  # "reasoning"
    reasoning: str


class ThinkingContentBlock(TypedDict, total=False):
    type: str  # "thinking"
    thinking: str
    signature: str


class GPT5ReasoningSummaryItem(TypedDict):
    type: str  # "summary_text"
    text: str


class GPT5ReasoningOutput(TypedDict):
    id: str
    type: str  # "reasoning"
    summary: list[GPT5ReasoningSummaryItem]


class ImageGenerationOutput(TypedDict, total=False):
    id: str
    type: str  # "image_generation_call"
    status: str
    result: str  # base64
    revised_prompt: str
    size: str
    output_format: str
    quality: str
    background: str


@dataclass
class MessageSeenEntry:
    text: bool = False
    reasoning: bool = False
    tool: dict[str, bool] = field(default_factory=dict)


@dataclass
class ToolCallInfo:
    id: str
    name: str


@dataclass
class LangGraphEventState:
    """State carried across LangGraph stream events.

    Mirrors `LangGraphEventState` in types.ts.
    """

    message_seen: dict[str, MessageSeenEntry] = field(default_factory=dict)
    """Tracks which message IDs have been seen with text/reasoning/tool flags."""

    message_concat: dict[str, Any] = field(default_factory=dict)
    """Accumulates AIMessageChunk-like dicts for later reference."""

    emitted_tool_calls: set[str] = field(default_factory=set)
    """Tool call IDs that have been emitted."""

    emitted_images: set[str] = field(default_factory=set)
    """Image IDs that have been emitted."""

    emitted_reasoning_ids: set[str] = field(default_factory=set)
    """Reasoning block IDs that have been emitted."""

    message_reasoning_ids: dict[str, str] = field(default_factory=dict)
    """Maps message IDs to their reasoning block IDs."""

    tool_call_info_by_index: dict[str, dict[int, ToolCallInfo]] = field(
        default_factory=dict
    )
    """Maps message ID → index → ToolCallInfo for streaming chunks without ID."""

    current_step: int | None = None
    """Tracks the current LangGraph step for start-step/finish-step events."""

    emitted_tool_calls_by_key: dict[str, str] = field(default_factory=dict)
    """Maps tool call key (name:argsJson) to tool call ID for HITL handling."""


@dataclass
class ModelStreamState:
    """State for model.astream() stream handling.

    Mirrors the inline `modelState` in adapter.ts:372–381.
    """

    started: bool = False
    message_id: str = "langchain-msg-1"
    reasoning_started: bool = False
    text_started: bool = False
    text_message_id: str | None = None
    reasoning_message_id: str | None = None
    emitted_images: set[str] = field(default_factory=set)
