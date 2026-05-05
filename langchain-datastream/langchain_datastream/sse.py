"""SSE framing for the AI SDK Data Stream Protocol.

Wire format per chunk:
    data: {json}\n\n

Stream ends with:
    data: [DONE]\n\n
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterable, AsyncIterator


def _fallback(value: Any) -> Any:
    """Last-resort serializer for objects that slipped through chunk coercion."""
    content = getattr(value, "content", None)
    if isinstance(content, (str, list, dict)):
        return content
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:
            pass
    return str(value)


def encode_sse_chunk(chunk: dict[str, Any]) -> str:
    return f"data: {json.dumps(chunk, separators=(',', ':'), default=_fallback)}\n\n"


def encode_sse_done() -> str:
    return "data: [DONE]\n\n"


async def frame_data_stream(
    chunks: AsyncIterable[dict[str, Any]],
) -> AsyncIterator[str]:
    """Wrap a chunk stream as SSE-encoded strings, terminated with `[DONE]`."""
    async for chunk in chunks:
        yield encode_sse_chunk(chunk)
    yield encode_sse_done()
