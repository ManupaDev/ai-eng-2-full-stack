"""Phase 4c smoke tests: process_langgraph_event."""

from langchain_datastream.langgraph_stream import process_langgraph_event
from langchain_datastream.types import LangGraphEventState


def _drive(events, state=None):
    out: list = []
    state = state or LangGraphEventState()
    for ev in events:
        process_langgraph_event(ev, state, out.append)
    return out, state


def test_custom_event_with_id_is_persistent():
    out, _ = _drive(
        [
            ("custom", {"type": "progress", "id": "p1", "value": 50}),
        ]
    )
    assert out == [
        {
            "type": "data-progress",
            "transient": False,
            "data": {"type": "progress", "id": "p1", "value": 50},
            "id": "p1",
        }
    ]


def test_custom_event_without_id_is_transient():
    out, _ = _drive([("custom", {"key": "value"})])
    assert out == [
        {
            "type": "data-custom",
            "transient": True,
            "data": {"key": "value"},
        }
    ]


def test_messages_text_chunk_emits_step_start_and_text():
    msg = {
        "type": "constructor",
        "id": ["langchain_core", "messages", "AIMessageChunk"],
        "kwargs": {"id": "run-1", "content": "hello"},
    }
    out, state = _drive(
        [("messages", [msg, {"langgraph_step": 1, "langgraph_node": "agent"}])]
    )
    assert out == [
        {"type": "start-step"},
        {"type": "text-start", "id": "run-1"},
        {"type": "text-delta", "delta": "hello", "id": "run-1"},
    ]
    assert state.current_step == 1
    assert state.message_seen["run-1"].text


def test_messages_tool_call_chunks_streaming():
    state = LangGraphEventState()
    out: list = []

    # First chunk: id + name, empty args
    msg1 = {
        "type": "constructor",
        "id": ["langchain_core", "messages", "AIMessageChunk"],
        "kwargs": {
            "id": "msg-1",
            "content": "",
            "tool_call_chunks": [
                {"index": 0, "id": "call_x", "name": "search", "args": ""}
            ],
        },
    }
    process_langgraph_event(
        ("messages", [msg1, {"langgraph_step": 1}]), state, out.append
    )

    # Second chunk: just args delta
    msg2 = {
        "type": "constructor",
        "id": ["langchain_core", "messages", "AIMessageChunk"],
        "kwargs": {
            "id": "msg-1",
            "content": "",
            "tool_call_chunks": [{"index": 0, "args": '{"q":'}],
        },
    }
    process_langgraph_event(
        ("messages", [msg2, {"langgraph_step": 1}]), state, out.append
    )

    msg3 = {
        "type": "constructor",
        "id": ["langchain_core", "messages", "AIMessageChunk"],
        "kwargs": {
            "id": "msg-1",
            "content": "",
            "tool_call_chunks": [{"index": 0, "args": '"x"}'}],
        },
    }
    process_langgraph_event(
        ("messages", [msg3, {"langgraph_step": 1}]), state, out.append
    )

    assert out == [
        {"type": "start-step"},
        {
            "type": "tool-input-start",
            "toolCallId": "call_x",
            "toolName": "search",
            "dynamic": True,
        },
        {"type": "tool-input-delta", "toolCallId": "call_x", "inputTextDelta": '{"q":'},
        {"type": "tool-input-delta", "toolCallId": "call_x", "inputTextDelta": '"x"}'},
    ]


def test_tool_message_emits_output_available():
    out, _ = _drive(
        [
            (
                "messages",
                [
                    {
                        "type": "constructor",
                        "id": ["langchain_core", "messages", "ToolMessage"],
                        "kwargs": {
                            "id": "tool-msg-1",
                            "tool_call_id": "call_x",
                            "content": "result text",
                        },
                    },
                    {"langgraph_step": 2},
                ],
            )
        ]
    )
    assert {"type": "tool-output-available", "toolCallId": "call_x", "output": "result text"} in out


def test_values_finalizes_open_text():
    state = LangGraphEventState()
    out: list = []
    msg = {
        "type": "constructor",
        "id": ["langchain_core", "messages", "AIMessageChunk"],
        "kwargs": {"id": "run-1", "content": "hi"},
    }
    process_langgraph_event(
        ("messages", [msg, {"langgraph_step": 1}]), state, out.append
    )
    out.clear()

    process_langgraph_event(
        ("values", {"messages": []}), state, out.append
    )
    assert out == [{"type": "text-end", "id": "run-1"}]
    assert "run-1" not in state.message_seen


def test_values_emits_tool_call_for_unstreamed_tool():
    """Tool call appears only in values.messages with no streamed chunks for it."""
    state = LangGraphEventState()
    out: list = []
    process_langgraph_event(
        (
            "values",
            {
                "messages": [
                    {
                        "type": "constructor",
                        "id": ["langchain_core", "messages", "AIMessage"],
                        "kwargs": {
                            "id": "msg-1",
                            "tool_calls": [
                                {"id": "call_1", "name": "search", "args": {"q": "x"}}
                            ],
                        },
                    }
                ]
            },
        ),
        state,
        out.append,
    )
    assert out == [
        {"type": "tool-input-start", "toolCallId": "call_1", "toolName": "search", "dynamic": True},
        {
            "type": "tool-input-available",
            "toolCallId": "call_1",
            "toolName": "search",
            "input": {"q": "x"},
            "dynamic": True,
        },
    ]


def test_hitl_interrupt_emits_approval_request(monkeypatch):
    from langchain_datastream import langgraph_stream

    monkeypatch.setattr(langgraph_stream, "_now_ms", lambda: 1234567890)

    state = LangGraphEventState()
    out: list = []
    process_langgraph_event(
        (
            "values",
            {
                "__interrupt__": [
                    {
                        "value": {
                            "action_requests": [
                                {"name": "delete_file", "args": {"f": "x.pdf"}}
                            ]
                        }
                    }
                ]
            },
        ),
        state,
        out.append,
    )
    expected_id = "hitl-delete_file-1234567890"
    assert out == [
        {
            "type": "tool-input-start",
            "toolCallId": expected_id,
            "toolName": "delete_file",
            "dynamic": True,
        },
        {
            "type": "tool-input-available",
            "toolCallId": expected_id,
            "toolName": "delete_file",
            "input": {"f": "x.pdf"},
            "dynamic": True,
        },
        {
            "type": "tool-approval-request",
            "approvalId": expected_id,
            "toolCallId": expected_id,
        },
    ]
