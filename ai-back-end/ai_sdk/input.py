"""Convert ModelMessage dicts to LangChain BaseMessage objects.

Mirrors `/tmp/ai-sdk-repo/packages/langchain/src/utils.ts:45-402` and
`/tmp/ai-sdk-repo/packages/langchain/src/adapter.ts:44-92`.
"""

from __future__ import annotations

import base64
import json
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .convert_to_model_messages import convert_to_model_messages
from .guards import is_tool_result_part


def _default_filename(media_type: str, prefix: str = "file") -> str:
    """utils.ts:113-119 — `image/png` → `file.png`."""
    parts = media_type.split("/")
    ext = parts[1] if len(parts) > 1 and parts[1] else "bin"
    return f"{prefix}.{ext}"


def _is_url_string(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _is_data_url(s: str) -> bool:
    return s.startswith("data:")


def _bytes_to_b64(b: bytes | bytearray) -> str:
    return base64.b64encode(bytes(b)).decode("ascii")


def _normalize_file_data(d: Any) -> Any:
    """Tagged shape `{type:'data',data}|{type:'url',url}|{type:'text',text}|{type:'reference',reference}`
    collapses to its inner value (matches utils.ts:239-261).
    """
    if isinstance(d, dict) and "type" in d:
        t = d.get("type")
        if t == "data":
            return d.get("data")
        if t == "url":
            return d.get("url")
        if t == "text":
            return d.get("text")
        # reference and unknowns: caller doesn't render — return ''
        return ""
    return d


def convert_tool_result_part(block: dict[str, Any]) -> ToolMessage:
    """utils.ts:45-73."""
    output = block.get("output") or {}
    out_type = output.get("type") if isinstance(output, dict) else None

    if out_type in ("text", "error-text"):
        content: Any = output.get("value", "")
    elif out_type in ("json", "error-json"):
        content = json.dumps(output.get("value"))
    elif out_type == "content":
        items = output.get("value", [])
        content = "".join(
            (item.get("text") if isinstance(item, dict) and item.get("type") == "text" else "")
            for item in items
        )
    else:
        content = ""

    return ToolMessage(content=content, tool_call_id=block.get("toolCallId", ""))


def convert_assistant_content(content: Any) -> AIMessage:
    """utils.ts:80-108."""
    if isinstance(content, str):
        return AIMessage(content=content)

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for part in content or []:
        if not isinstance(part, dict):
            continue
        t = part.get("type")
        if t == "text":
            text_parts.append(part.get("text", ""))
        elif t == "tool-call":
            tool_calls.append(
                {
                    "id": part.get("toolCallId", ""),
                    "name": part.get("toolName", ""),
                    "args": part.get("input", {}) or {},
                }
            )

    kwargs: dict[str, Any] = {"content": "".join(text_parts)}
    if tool_calls:
        kwargs["tool_calls"] = tool_calls
    return AIMessage(**kwargs)


def convert_user_content(content: Any) -> HumanMessage:
    """utils.ts:146-402."""
    if isinstance(content, str):
        return HumanMessage(content=content)

    blocks: list[dict[str, Any]] = []

    for part in content or []:
        if not isinstance(part, dict):
            continue
        t = part.get("type")
        if t == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif t == "image":
            _append_image_part(blocks, part)
        elif t == "file":
            _append_file_part(blocks, part)

    # If only text blocks, collapse to string for efficiency (utils.ts:393-399)
    if blocks and all(b.get("type") == "text" for b in blocks):
        return HumanMessage(content="".join(b.get("text", "") for b in blocks))
    return HumanMessage(content=blocks)


def _append_image_part(blocks: list[dict[str, Any]], part: dict[str, Any]) -> None:
    """utils.ts:156-221."""
    image = part.get("image")
    media_type = part.get("mediaType")

    # Python doesn't have a URL class; treat dict {type:'url',url} as URL
    if isinstance(image, dict) and image.get("type") == "url":
        blocks.append(
            {"type": "image_url", "image_url": {"url": str(image.get("url", ""))}}
        )
    elif isinstance(image, str):
        if _is_url_string(image) or _is_data_url(image):
            blocks.append({"type": "image_url", "image_url": {"url": image}})
        else:
            mime = media_type or "image/png"
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{image}"},
                }
            )
    elif isinstance(image, (bytes, bytearray)):
        b64 = _bytes_to_b64(image)
        mime = media_type or "image/png"
        blocks.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        )


def _append_file_part(blocks: list[dict[str, Any]], part: dict[str, Any]) -> None:
    """utils.ts:222-388."""
    raw_data = part.get("data")
    media_type = part.get("mediaType", "")
    filename = part.get("filename")

    data = _normalize_file_data(raw_data)

    is_image = isinstance(media_type, str) and media_type.startswith("image/")

    if is_image:
        if isinstance(data, str):
            if _is_url_string(data) or _is_data_url(data):
                blocks.append({"type": "image_url", "image_url": {"url": data}})
            else:
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    }
                )
        elif isinstance(data, (bytes, bytearray)):
            b64 = _bytes_to_b64(data)
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}"},
                }
            )
        elif isinstance(data, dict) and data.get("type") == "url":
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": str(data.get("url", ""))},
                }
            )
        return

    name = filename or _default_filename(media_type, "file")

    if isinstance(data, str):
        if _is_url_string(data):
            blocks.append(
                {
                    "type": "file",
                    "url": data,
                    "mimeType": media_type,
                    "filename": name,
                }
            )
        elif _is_data_url(data):
            # `data:<mime>;base64,<payload>` → split into payload + mime
            try:
                header, payload = data.split(",", 1)
                mime = header[len("data:") :].split(";", 1)[0]
                if ";base64" in header:
                    blocks.append(
                        {
                            "type": "file",
                            "data": payload,
                            "mimeType": mime,
                            "filename": name,
                        }
                    )
                else:
                    blocks.append(
                        {
                            "type": "file",
                            "url": data,
                            "mimeType": media_type,
                            "filename": name,
                        }
                    )
            except ValueError:
                blocks.append(
                    {
                        "type": "file",
                        "url": data,
                        "mimeType": media_type,
                        "filename": name,
                    }
                )
        else:
            blocks.append(
                {
                    "type": "file",
                    "data": data,
                    "mimeType": media_type,
                    "filename": name,
                }
            )
    elif isinstance(data, (bytes, bytearray)):
        b64 = _bytes_to_b64(data)
        blocks.append(
            {"type": "file", "data": b64, "mimeType": media_type, "filename": name}
        )
    elif isinstance(data, dict) and data.get("type") == "url":
        blocks.append(
            {
                "type": "file",
                "url": str(data.get("url", "")),
                "mimeType": media_type,
                "filename": name,
            }
        )


def convert_model_messages(messages: list[dict[str, Any]]) -> list[BaseMessage]:
    """adapter.ts:57-92."""
    result: list[BaseMessage] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role == "tool":
            for item in content or []:
                if is_tool_result_part(item):
                    result.append(convert_tool_result_part(item))
        elif role == "assistant":
            result.append(convert_assistant_content(content))
        elif role == "system":
            from langchain_core.messages import SystemMessage

            result.append(SystemMessage(content=content))
        elif role == "user":
            result.append(convert_user_content(content))
    return result


async def to_base_messages(messages: list[Any]) -> list[BaseMessage]:
    """adapter.ts:44-49 — UIMessage[] → BaseMessage[] via convert_to_model_messages."""
    model_messages = await convert_to_model_messages(messages)
    return convert_model_messages(model_messages)
