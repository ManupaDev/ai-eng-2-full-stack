"""FastAPI integration for the AI SDK Data Stream Protocol.

`ui_message_stream_response` wraps `to_ui_message_stream` output in an SSE-framed
``StreamingResponse`` with the headers Vercel `useChat` expects.
"""

from __future__ import annotations

from typing import Any, AsyncIterable

from fastapi.responses import StreamingResponse

from .adapter import to_ui_message_stream
from .callbacks import StreamCallbacks
from .sse import frame_data_stream

DATA_STREAM_HEADERS = {
    "x-vercel-ai-ui-message-stream": "v1",
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}


def ui_message_stream_response(
    stream: Any,
    *,
    callbacks: StreamCallbacks | None = None,
    extra_headers: dict[str, str] | None = None,
) -> StreamingResponse:
    """Build a FastAPI streaming response that emits AI SDK Data Stream Protocol over SSE.

    `stream` is whatever LangChain returns: `model.astream()`, `graph.astream(stream_mode=...)`,
    or `agent.astream_events(version='v2')`. The adapter detects the shape on the
    first item.
    """
    chunk_iter = to_ui_message_stream(stream, callbacks=callbacks)
    sse_iter = frame_data_stream(chunk_iter)
    headers = dict(DATA_STREAM_HEADERS)
    if extra_headers:
        headers.update(extra_headers)
    return StreamingResponse(
        sse_iter,
        media_type="text/event-stream",
        headers=headers,
    )
