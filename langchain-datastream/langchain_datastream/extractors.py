"""Content extractors for LangChain messages.

Mirrors `/tmp/ai-sdk-repo/packages/langchain/src/utils.ts:573-991` (subset that extracts
content). All functions accept either a class instance, a plain dict (RemoteGraph format),
or a serialized LangChain message dict (`{type: 'constructor', kwargs: {...}}`).
"""

from __future__ import annotations

from typing import Any

from .guards import (
    is_gpt5_reasoning_output,
    is_image_generation_output,
    is_reasoning_content_block,
    is_thinking_content_block,
)


def _kwargs(msg: Any) -> Any:
    """Resolve the data source from a message: kwargs for serialized, msg itself otherwise."""
    if isinstance(msg, dict):
        if msg.get("type") == "constructor" and isinstance(msg.get("kwargs"), dict):
            return msg["kwargs"]
        return msg
    return None


def _attr(msg: Any, name: str, default: Any = None) -> Any:
    """Read attr from a class instance OR a kwargs dict."""
    if isinstance(msg, dict):
        return msg.get(name, default)
    return getattr(msg, name, default)


def get_message_text(msg: Any) -> str:
    """utils.ts:680-723."""
    # Class instance with `.text` accessor (AIMessageChunk has it)
    text_attr = getattr(msg, "text", None)
    if text_attr is not None and not isinstance(text_attr, dict) and not callable(text_attr):
        if isinstance(text_attr, str):
            return text_attr

    if msg is None:
        return ""

    src = _kwargs(msg)
    if src is None:
        # Class instance fallback
        content = getattr(msg, "content", None)
    else:
        content = src.get("content")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
    return ""


def get_message_id(msg: Any) -> str | None:
    """utils.ts:573-600."""
    if msg is None:
        return None
    if isinstance(msg, dict):
        # Serialized form: id at kwargs.id
        if msg.get("type") == "constructor" and isinstance(msg.get("kwargs"), dict):
            kw_id = msg["kwargs"].get("id")
            return kw_id if isinstance(kw_id, str) else None
        # Plain dict
        msg_id = msg.get("id")
        return msg_id if isinstance(msg_id, str) else None
    # Class instance
    msg_id = getattr(msg, "id", None)
    return msg_id if isinstance(msg_id, str) else None


def _resolve_kwargs(msg: Any) -> dict[str, Any]:
    """Get the dict-like data source: kwargs for serialized, the dict itself for plain, attrs for instance."""
    if isinstance(msg, dict):
        if msg.get("type") == "constructor" and isinstance(msg.get("kwargs"), dict):
            return msg["kwargs"]
        return msg
    # Class instance — pull additional_kwargs and response_metadata if available
    return {
        "additional_kwargs": getattr(msg, "additional_kwargs", None),
        "response_metadata": getattr(msg, "response_metadata", None),
        "content": getattr(msg, "content", None),
    }


def extract_reasoning_id(msg: Any) -> str | None:
    """utils.ts:785-816."""
    if msg is None:
        return None
    kw = _resolve_kwargs(msg)

    # additional_kwargs.reasoning.id (GPT-5 streaming)
    ak = kw.get("additional_kwargs")
    if isinstance(ak, dict):
        reasoning = ak.get("reasoning")
        if isinstance(reasoning, dict) and isinstance(reasoning.get("id"), str):
            return reasoning["id"]

    # response_metadata.output[].id where output item is GPT-5 reasoning
    rm = kw.get("response_metadata")
    if isinstance(rm, dict):
        output = rm.get("output")
        if isinstance(output, list):
            for item in output:
                if is_gpt5_reasoning_output(item):
                    item_id = item.get("id") if isinstance(item, dict) else None
                    if isinstance(item_id, str):
                        return item_id
    return None


def extract_reasoning_from_content_blocks(msg: Any) -> str | None:
    """utils.ts:832-887.

    For STREAMING chunks. Reads delta-based content from contentBlocks (Anthropic)
    or additional_kwargs.reasoning.summary (GPT-5).
    """
    if msg is None:
        return None
    kw = _resolve_kwargs(msg)

    # contentBlocks (Anthropic)
    cb = kw.get("contentBlocks")
    if isinstance(cb, list):
        parts: list[str] = []
        for block in cb:
            if is_reasoning_content_block(block):
                parts.append(block.get("reasoning", ""))
            elif is_thinking_content_block(block):
                parts.append(block.get("thinking", ""))
        if parts:
            return "".join(parts)

    # additional_kwargs.reasoning.summary[].text (GPT-5 streaming)
    ak = kw.get("additional_kwargs")
    if isinstance(ak, dict):
        reasoning = ak.get("reasoning")
        if isinstance(reasoning, dict):
            summary = reasoning.get("summary")
            if isinstance(summary, list):
                parts = [
                    item["text"]
                    for item in summary
                    if isinstance(item, dict) and isinstance(item.get("text"), str)
                ]
                if parts:
                    return "".join(parts)
    return None


def extract_reasoning_from_values_message(msg: Any) -> str | None:
    """utils.ts:896-957.

    For VALUES events. Reads accumulated reasoning from response_metadata.output
    (GPT-5 final), with additional_kwargs.reasoning.summary as fallback.
    """
    if msg is None:
        return None
    kw = _resolve_kwargs(msg)

    rm = kw.get("response_metadata")
    if isinstance(rm, dict):
        output = rm.get("output")
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                if is_gpt5_reasoning_output(item) and isinstance(item, dict):
                    for s in item.get("summary", []):
                        if isinstance(s, dict) and isinstance(s.get("text"), str) and s["text"]:
                            parts.append(s["text"])
            if parts:
                return "".join(parts)

    ak = kw.get("additional_kwargs")
    if isinstance(ak, dict):
        reasoning = ak.get("reasoning")
        if isinstance(reasoning, dict):
            summary = reasoning.get("summary")
            if isinstance(summary, list):
                parts = [
                    s["text"]
                    for s in summary
                    if isinstance(s, dict) and isinstance(s.get("text"), str)
                ]
                if parts:
                    return "".join(parts)
    return None


def extract_image_outputs(additional_kwargs: dict[str, Any] | None) -> list[dict[str, Any]]:
    """utils.ts:982-991."""
    if not additional_kwargs:
        return []
    tool_outputs = additional_kwargs.get("tool_outputs")
    if not isinstance(tool_outputs, list):
        return []
    return [t for t in tool_outputs if is_image_generation_output(t)]
