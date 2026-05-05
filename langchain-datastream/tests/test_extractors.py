"""Phase 3 verification: content extractors."""

from langchain_core.messages import AIMessageChunk

from langchain_datastream.extractors import (
    extract_image_outputs,
    extract_reasoning_from_content_blocks,
    extract_reasoning_from_values_message,
    extract_reasoning_id,
    get_message_id,
    get_message_text,
)


# ---- get_message_text ----


def test_get_message_text_class_instance_text_attr():
    assert get_message_text(AIMessageChunk(content="hello")) == "hello"


def test_get_message_text_dict_string_content():
    assert get_message_text({"content": "hi"}) == "hi"


def test_get_message_text_dict_block_content():
    assert (
        get_message_text(
            {
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "image", "url": "x"},
                    {"type": "text", "text": "b"},
                ]
            }
        )
        == "ab"
    )


def test_get_message_text_serialized():
    assert (
        get_message_text(
            {"type": "constructor", "id": ["x"], "kwargs": {"content": "wrapped"}}
        )
        == "wrapped"
    )


def test_get_message_text_none_or_unknown():
    assert get_message_text(None) == ""
    assert get_message_text({}) == ""


# ---- get_message_id ----


def test_get_message_id_dict():
    assert get_message_id({"id": "abc"}) == "abc"


def test_get_message_id_serialized():
    assert (
        get_message_id(
            {"type": "constructor", "id": ["x"], "kwargs": {"id": "wrapped"}}
        )
        == "wrapped"
    )


def test_get_message_id_missing():
    assert get_message_id({}) is None
    assert get_message_id(None) is None


# ---- extract_reasoning_id ----


def test_extract_reasoning_id_streaming():
    msg = {"additional_kwargs": {"reasoning": {"id": "rid-1"}}}
    assert extract_reasoning_id(msg) == "rid-1"


def test_extract_reasoning_id_from_response_metadata():
    msg = {
        "response_metadata": {
            "output": [
                {"type": "reasoning", "id": "r-final", "summary": [{"text": "x"}]}
            ]
        }
    }
    assert extract_reasoning_id(msg) == "r-final"


def test_extract_reasoning_id_serialized():
    msg = {
        "type": "constructor",
        "id": ["x"],
        "kwargs": {"additional_kwargs": {"reasoning": {"id": "rid-w"}}},
    }
    assert extract_reasoning_id(msg) == "rid-w"


def test_extract_reasoning_id_missing():
    assert extract_reasoning_id({}) is None


# ---- extract_reasoning_from_content_blocks (streaming) ----


def test_extract_reasoning_from_anthropic_content_blocks():
    msg = {
        "contentBlocks": [
            {"type": "reasoning", "reasoning": "step1 "},
            {"type": "text", "text": "ignored"},
            {"type": "thinking", "thinking": "step2"},
        ]
    }
    assert extract_reasoning_from_content_blocks(msg) == "step1 step2"


def test_extract_reasoning_from_gpt5_streaming_summary():
    msg = {
        "additional_kwargs": {
            "reasoning": {
                "summary": [
                    {"type": "summary_text", "text": "**Plan"},
                    {"text": "ning**"},
                ]
            }
        }
    }
    assert extract_reasoning_from_content_blocks(msg) == "**Planning**"


def test_extract_reasoning_from_content_blocks_missing():
    assert extract_reasoning_from_content_blocks({}) is None


# ---- extract_reasoning_from_values_message (final/values) ----


def test_extract_reasoning_from_values_response_metadata():
    msg = {
        "response_metadata": {
            "output": [
                {
                    "type": "reasoning",
                    "id": "r1",
                    "summary": [
                        {"type": "summary_text", "text": "all "},
                        {"text": "done"},
                    ],
                }
            ]
        }
    }
    assert extract_reasoning_from_values_message(msg) == "all done"


def test_extract_reasoning_from_values_fallback_to_additional_kwargs():
    msg = {"additional_kwargs": {"reasoning": {"summary": [{"text": "fallback"}]}}}
    assert extract_reasoning_from_values_message(msg) == "fallback"


def test_extract_reasoning_from_values_missing():
    assert extract_reasoning_from_values_message({}) is None


# ---- extract_image_outputs ----


def test_extract_image_outputs():
    ak = {
        "tool_outputs": [
            {"type": "image_generation_call", "id": "i1", "result": "AAAA"},
            {"type": "code_interpreter_call", "id": "c1"},
            {"type": "image_generation_call", "id": "i2", "result": "BBBB"},
        ]
    }
    out = extract_image_outputs(ak)
    assert len(out) == 2
    assert {o["id"] for o in out} == {"i1", "i2"}


def test_extract_image_outputs_empty():
    assert extract_image_outputs(None) == []
    assert extract_image_outputs({}) == []
    assert extract_image_outputs({"tool_outputs": "not-a-list"}) == []
