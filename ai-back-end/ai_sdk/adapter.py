"""Top-level stream orchestrator.

Mirrors `/tmp/ai-sdk-repo/packages/langchain/src/adapter.ts:357-587` (`toUIMessageStream`).

Detects which of three input shapes is being streamed and routes through the
correct processor:
  - dict with `event` field            → astream_events(version='v2')
  - list/tuple                         → graph.astream(stream_mode=['values','messages'])
  - else (AIMessageChunk-like)         → model.astream()
"""

from __future__ import annotations

from asyncio import CancelledError
from typing import Any, AsyncIterator, Iterable

from .callbacks import StreamCallbacks, maybe_await
from .langgraph_stream import process_langgraph_event, parse_langgraph_event
from .model_stream import process_model_chunk
from .stream_events import process_stream_events_event
from .types import LangGraphEventState, ModelStreamState


def _is_stream_events_event(value: Any) -> bool:
    """utils.ts:101-112 — `{event: str, data: any}`."""
    return (
        isinstance(value, dict)
        and isinstance(value.get("event"), str)
        and "data" in value
    )


def _is_langgraph_event(value: Any) -> bool:
    return isinstance(value, (list, tuple))


async def _to_async_iter(stream: Any) -> AsyncIterator[Any]:
    """Accept either an async iterable or a sync iterable; yield items async."""
    if hasattr(stream, "__aiter__"):
        async for item in stream:
            yield item
        return
    if isinstance(stream, Iterable):
        for item in stream:
            yield item
        return
    raise TypeError(f"Stream is neither async nor sync iterable: {stream!r}")


async def to_ui_message_stream(
    stream: Any,
    *,
    callbacks: StreamCallbacks | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """adapter.ts:357-587.

    Yields UIMessageChunk dicts (camelCase JSON shape). The caller wraps these
    in SSE framing for the wire.
    """
    text_chunks: list[str] = []
    last_values_data: Any = None

    model_state = ModelStreamState()
    langgraph_state = LangGraphEventState()
    stream_type: str | None = None  # 'model' | 'langgraph' | 'streamEvents'

    if callbacks and callbacks.on_start:
        await maybe_await(callbacks.on_start())

    yield {"type": "start"}

    try:
        async for value in _to_async_iter(stream):
            if stream_type is None:
                if _is_langgraph_event(value):
                    stream_type = "langgraph"
                elif _is_stream_events_event(value):
                    stream_type = "streamEvents"
                else:
                    stream_type = "model"

            buffer: list[dict[str, Any]] = []

            def emit(chunk: dict[str, Any]) -> None:
                buffer.append(chunk)

            if stream_type == "model":
                process_model_chunk(value, model_state, emit)
            elif stream_type == "streamEvents":
                process_stream_events_event(value, model_state, emit)
            else:
                kind, payload = parse_langgraph_event(value)
                if kind == "values":
                    last_values_data = payload
                process_langgraph_event(value, langgraph_state, emit)

            for chunk in buffer:
                if (
                    callbacks
                    and chunk.get("type") == "text-delta"
                    and isinstance(chunk.get("delta"), str)
                ):
                    text_chunks.append(chunk["delta"])
                    if callbacks.on_token:
                        await maybe_await(callbacks.on_token(chunk["delta"]))
                    if callbacks.on_text:
                        await maybe_await(callbacks.on_text(chunk["delta"]))
                yield chunk

        # Finalize (adapter.ts:520-559)
        if stream_type in ("model", "streamEvents"):
            if model_state.reasoning_started:
                yield {
                    "type": "reasoning-end",
                    "id": model_state.reasoning_message_id or model_state.message_id,
                }
            if model_state.text_started:
                yield {
                    "type": "text-end",
                    "id": model_state.text_message_id or model_state.message_id,
                }
            yield {"type": "finish"}
        elif stream_type == "langgraph":
            for id_, seen in list(langgraph_state.message_seen.items()):
                if seen.text:
                    yield {"type": "text-end", "id": id_}
                if seen.reasoning:
                    yield {"type": "reasoning-end", "id": id_}
            if langgraph_state.current_step is not None:
                yield {"type": "finish-step"}
            yield {"type": "finish"}
        else:
            # Empty stream: still emit finish for client clean-up.
            yield {"type": "finish"}

        if callbacks:
            if callbacks.on_final:
                await maybe_await(callbacks.on_final("".join(text_chunks)))
            if callbacks.on_finish:
                await maybe_await(callbacks.on_finish(last_values_data))

    except BaseException as exc:
        if callbacks and callbacks.on_final:
            await maybe_await(callbacks.on_final("".join(text_chunks)))

        is_abort = isinstance(exc, (CancelledError, GeneratorExit)) or (
            type(exc).__name__ in ("AbortError",)
        )
        if callbacks:
            if is_abort and callbacks.on_abort:
                await maybe_await(callbacks.on_abort())
            elif callbacks.on_error:
                err = exc if isinstance(exc, Exception) else Exception(str(exc))
                await maybe_await(callbacks.on_error(err))

        yield {"type": "error", "errorText": str(exc)}
