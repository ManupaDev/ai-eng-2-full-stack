"""Phase 1: type guard parity vs utils.ts."""

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from langchain_datastream.guards import (
    is_ai_message_chunk,
    is_gpt5_reasoning_output,
    is_image_generation_output,
    is_reasoning_content_block,
    is_thinking_content_block,
    is_tool_message_type,
    is_tool_result_part,
)


def test_is_tool_result_part():
    assert is_tool_result_part({"type": "tool-result", "toolCallId": "x"})
    assert not is_tool_result_part({"type": "text", "text": "hi"})
    assert not is_tool_result_part(None)
    assert not is_tool_result_part("string")


def test_is_ai_message_chunk_class_instance():
    assert is_ai_message_chunk(AIMessageChunk(content="hi"))


def test_is_ai_message_chunk_remote_graph_format():
    assert is_ai_message_chunk({"type": "ai", "content": "hi"})


def test_is_ai_message_chunk_serialized():
    serialized = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain_core", "messages", "AIMessageChunk"],
        "kwargs": {"content": "hi"},
    }
    assert is_ai_message_chunk(serialized)


def test_is_ai_message_chunk_negative():
    assert not is_ai_message_chunk(HumanMessage(content="hi"))
    assert not is_ai_message_chunk({"type": "human"})
    assert not is_ai_message_chunk(None)


def test_is_tool_message_type():
    assert is_tool_message_type(ToolMessage(content="x", tool_call_id="c"))
    assert is_tool_message_type({"type": "tool", "tool_call_id": "c"})
    assert is_tool_message_type(
        {
            "type": "constructor",
            "id": ["langchain_core", "messages", "ToolMessage"],
            "kwargs": {},
        }
    )
    assert not is_tool_message_type(AIMessage(content="hi"))


def test_is_reasoning_content_block():
    assert is_reasoning_content_block({"type": "reasoning", "reasoning": "hmm"})
    assert not is_reasoning_content_block({"type": "reasoning", "reasoning": 42})
    assert not is_reasoning_content_block({"type": "text", "text": "hi"})


def test_is_thinking_content_block():
    assert is_thinking_content_block({"type": "thinking", "thinking": "hmm"})
    assert not is_thinking_content_block({"type": "thinking"})


def test_is_gpt5_reasoning_output():
    assert is_gpt5_reasoning_output(
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "x"}]}
    )
    assert not is_gpt5_reasoning_output({"type": "reasoning", "reasoning": "x"})


def test_is_image_generation_output():
    assert is_image_generation_output({"type": "image_generation_call", "id": "x"})
    assert not is_image_generation_output({"type": "tool_call"})
