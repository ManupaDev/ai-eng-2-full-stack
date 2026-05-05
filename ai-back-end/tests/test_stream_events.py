"""Phase 4b verification: process_stream_events_event."""

from ai_sdk.stream_events import process_stream_events_event
from ai_sdk.types import ModelStreamState


def test_chat_model_start_captures_run_id():
    state = ModelStreamState()
    out: list = []
    process_stream_events_event(
        {"event": "on_chat_model_start", "run_id": "run-x", "data": {}},
        state,
        out.append,
    )
    assert state.message_id == "run-x"
    assert out == []


def test_chat_model_stream_emits_text_lifecycle():
    state = ModelStreamState()
    out: list = []
    process_stream_events_event(
        {
            "event": "on_chat_model_stream",
            "run_id": "run-x",
            "data": {"chunk": {"id": "run-x", "content": "hi"}},
        },
        state,
        out.append,
    )
    assert out == [
        {"type": "text-start", "id": "run-x"},
        {"type": "text-delta", "delta": "hi", "id": "run-x"},
    ]


def test_chat_model_stream_reasoning_then_text_closes_reasoning():
    state = ModelStreamState()
    out: list = []
    process_stream_events_event(
        {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": {
                    "id": "r1",
                    "content": "",
                    "additional_kwargs": {
                        "reasoning": {
                            "summary": [{"type": "summary_text", "text": "thoughts"}]
                        }
                    },
                }
            },
        },
        state,
        out.append,
    )
    assert out == [
        {"type": "reasoning-start", "id": "r1"},
        {"type": "reasoning-delta", "delta": "thoughts", "id": "r1"},
    ]

    out.clear()
    process_stream_events_event(
        {
            "event": "on_chat_model_stream",
            "data": {"chunk": {"id": "r1", "content": "answer"}},
        },
        state,
        out.append,
    )
    assert out == [
        {"type": "reasoning-end", "id": "r1"},
        {"type": "text-start", "id": "r1"},
        {"type": "text-delta", "delta": "answer", "id": "r1"},
    ]


def test_tool_start_emits_dynamic_tool_input_start():
    state = ModelStreamState()
    out: list = []
    process_stream_events_event(
        {
            "event": "on_tool_start",
            "run_id": "tool-1",
            "name": "search",
            "data": {"input": {"q": "x"}},
        },
        state,
        out.append,
    )
    assert out == [
        {
            "type": "tool-input-start",
            "toolCallId": "tool-1",
            "toolName": "search",
            "dynamic": True,
        }
    ]


def test_tool_end_emits_output_available():
    state = ModelStreamState()
    out: list = []
    process_stream_events_event(
        {
            "event": "on_tool_end",
            "run_id": "tool-1",
            "data": {"output": "result"},
        },
        state,
        out.append,
    )
    assert out == [
        {"type": "tool-output-available", "toolCallId": "tool-1", "output": "result"}
    ]


def test_data_none_is_skipped():
    state = ModelStreamState()
    out: list = []
    process_stream_events_event(
        {"event": "on_chat_model_stream", "data": None}, state, out.append
    )
    assert out == []
