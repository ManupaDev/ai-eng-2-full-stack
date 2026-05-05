"""Type guards for LangChain messages and content blocks.

Mirrors `/tmp/ai-sdk-repo/packages/langchain/src/utils.ts:409–991`.

Each guard handles three input shapes:
1. LangChain class instances (e.g. AIMessageChunk from langchain_core.messages)
2. Plain dicts with `type` field (RemoteGraph format)
3. Serialized LangChain dicts with `type: "constructor"`, `id: [...path]`, `kwargs: {...}`
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.messages import AIMessageChunk, ToolMessage
except ImportError:  # pragma: no cover
    AIMessageChunk = None  # type: ignore[assignment,misc]
    ToolMessage = None  # type: ignore[assignment,misc]


def is_tool_result_part(item: Any) -> bool:
    """utils.ts:409 — checks for ToolResultPart shape."""
    return (
        item is not None
        and isinstance(item, dict)
        and item.get("type") == "tool-result"
    )


def is_plain_message_object(msg: Any) -> bool:
    """utils.ts:558 — checks if msg is a plain dict (not a LangChain class instance).

    LangChain class instances expose a ``_get_type`` method (Python equivalent of TS ``_getType``).
    """
    if msg is None or not isinstance(msg, (dict, object)):
        return False
    if isinstance(msg, dict):
        return True
    # Class instance check: LangChain BaseMessage subclasses define _get_type() / type
    return not callable(getattr(msg, "_get_type", None)) and not callable(
        getattr(msg, "_getType", None)
    )


def _is_class_instance(msg: Any, cls: Any) -> bool:
    """Helper: is `msg` an instance of LangChain `cls` (when cls is importable)."""
    return cls is not None and isinstance(msg, cls)


def _has_constructor_path(msg: dict[str, Any], names: tuple[str, ...]) -> bool:
    """Helper: serialized LangChain message has `type: 'constructor'` and one of `names` in its `id` path."""
    if msg.get("type") != "constructor":
        return False
    msg_id = msg.get("id")
    if not isinstance(msg_id, list):
        return False
    return any(n in msg_id for n in names)


def is_ai_message_chunk(msg: Any) -> bool:
    """utils.ts:611 — accepts class instance, plain dict with `type='ai'`, or serialized form."""
    if _is_class_instance(msg, AIMessageChunk):
        return True
    if isinstance(msg, dict):
        if msg.get("type") == "ai":
            return True
        if _has_constructor_path(msg, ("AIMessageChunk", "AIMessage")):
            return True
    return False


def is_tool_message_type(msg: Any) -> bool:
    """utils.ts:647 — accepts ToolMessage instance, plain dict with `type='tool'`, or serialized form."""
    if _is_class_instance(msg, ToolMessage):
        return True
    if isinstance(msg, dict):
        if msg.get("type") == "tool":
            return True
        if _has_constructor_path(msg, ("ToolMessage",)):
            return True
    return False


def is_reasoning_content_block(obj: Any) -> bool:
    """utils.ts:731 — `{type: 'reasoning', reasoning: str}`."""
    return (
        obj is not None
        and isinstance(obj, dict)
        and obj.get("type") == "reasoning"
        and isinstance(obj.get("reasoning"), str)
    )


def is_thinking_content_block(obj: Any) -> bool:
    """utils.ts:750 — `{type: 'thinking', thinking: str}` (Anthropic-style)."""
    return (
        obj is not None
        and isinstance(obj, dict)
        and obj.get("type") == "thinking"
        and isinstance(obj.get("thinking"), str)
    )


def is_gpt5_reasoning_output(obj: Any) -> bool:
    """utils.ts:766 — `{type: 'reasoning', summary: list}`."""
    return (
        obj is not None
        and isinstance(obj, dict)
        and obj.get("type") == "reasoning"
        and isinstance(obj.get("summary"), list)
    )


def is_image_generation_output(obj: Any) -> bool:
    """utils.ts:965 — `{type: 'image_generation_call', ...}`."""
    return (
        obj is not None
        and isinstance(obj, dict)
        and obj.get("type") == "image_generation_call"
    )
