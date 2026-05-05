"""Phase 5a: to_ui_message_stream end-to-end."""

import pytest

from ai_sdk import StreamCallbacks, to_ui_message_stream
from ai_sdk.sse import frame_data_stream

pytestmark = pytest.mark.asyncio


async def _collect(stream):
    return [c async for c in stream]


async def _aiter(items):
    for item in items:
        yield item


# ---------- model stream path ----------


async def test_model_stream_text_lifecycle():
    chunks = await _collect(
        to_ui_message_stream(_aiter([{"id": "m-1", "content": "hi"}]))
    )
    assert chunks == [
        {"type": "start"},
        {"type": "text-start", "id": "m-1"},
        {"type": "text-delta", "delta": "hi", "id": "m-1"},
        {"type": "text-end", "id": "m-1"},
        {"type": "finish"},
    ]


# ---------- streamEvents path ----------


async def test_stream_events_text_lifecycle():
    chunks = await _collect(
        to_ui_message_stream(
            _aiter(
                [
                    {
                        "event": "on_chat_model_stream",
                        "run_id": "r-1",
                        "data": {"chunk": {"id": "r-1", "content": "yo"}},
                    }
                ]
            )
        )
    )
    assert chunks == [
        {"type": "start"},
        {"type": "text-start", "id": "r-1"},
        {"type": "text-delta", "delta": "yo", "id": "r-1"},
        {"type": "text-end", "id": "r-1"},
        {"type": "finish"},
    ]


async def test_stream_events_tool_call():
    chunks = await _collect(
        to_ui_message_stream(
            _aiter(
                [
                    {
                        "event": "on_tool_start",
                        "run_id": "tool-1",
                        "name": "search",
                        "data": {"input": {"q": "x"}},
                    },
                    {
                        "event": "on_tool_end",
                        "run_id": "tool-1",
                        "data": {"output": "result"},
                    },
                ]
            )
        )
    )
    assert {
        "type": "tool-input-start",
        "toolCallId": "tool-1",
        "toolName": "search",
        "dynamic": True,
    } in chunks
    assert {
        "type": "tool-output-available",
        "toolCallId": "tool-1",
        "output": "result",
    } in chunks
    assert chunks[0] == {"type": "start"}
    assert chunks[-1] == {"type": "finish"}


# ---------- langgraph path ----------


async def test_langgraph_text_then_values():
    msg = {
        "type": "constructor",
        "id": ["langchain_core", "messages", "AIMessageChunk"],
        "kwargs": {"id": "lg-1", "content": "hello"},
    }
    chunks = await _collect(
        to_ui_message_stream(
            _aiter(
                [
                    ("messages", [msg, {"langgraph_step": 1}]),
                    ("values", {"messages": []}),
                ]
            )
        )
    )
    # Expect: start, start-step, text-start, text-delta, text-end, finish-step, finish
    assert chunks[0] == {"type": "start"}
    assert {"type": "start-step"} in chunks
    assert {"type": "text-start", "id": "lg-1"} in chunks
    assert {"type": "text-delta", "delta": "hello", "id": "lg-1"} in chunks
    assert {"type": "text-end", "id": "lg-1"} in chunks
    assert {"type": "finish-step"} in chunks
    assert chunks[-1] == {"type": "finish"}


# ---------- callbacks ----------


async def test_callbacks_fire():
    seen_tokens = []
    seen_text = []
    seen_final = []
    seen_started = []

    def on_start():
        seen_started.append(True)

    async def on_token(t):
        seen_tokens.append(t)

    def on_text(t):
        seen_text.append(t)

    async def on_final(t):
        seen_final.append(t)

    cb = StreamCallbacks(
        on_start=on_start, on_token=on_token, on_text=on_text, on_final=on_final
    )
    await _collect(
        to_ui_message_stream(_aiter([{"id": "x", "content": "abc"}]), callbacks=cb)
    )
    assert seen_started == [True]
    assert seen_tokens == ["abc"]
    assert seen_text == ["abc"]
    assert seen_final == ["abc"]


# ---------- sse framing ----------


async def test_sse_framing_emits_done_terminator():
    sse = await _collect(
        frame_data_stream(to_ui_message_stream(_aiter([{"id": "x", "content": "ok"}])))
    )
    assert sse[0] == 'data: {"type":"start"}\n\n'
    assert sse[-1] == "data: [DONE]\n\n"
