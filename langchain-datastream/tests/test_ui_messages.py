"""Phase 2a verification: UIMessage parts + guards."""

import pytest

from langchain_datastream.ui_messages import (
    DataUIPart,
    DynamicToolUIPart,
    FileUIPart,
    ReasoningUIPart,
    StepStartUIPart,
    TextUIPart,
    ToolUIPart,
    UIMessage,
    get_static_tool_name,
    get_tool_name,
    is_data_ui_part,
    is_dynamic_tool_ui_part,
    is_file_ui_part,
    is_reasoning_ui_part,
    is_static_tool_ui_part,
    is_step_start_ui_part,
    is_text_ui_part,
    is_tool_ui_part,
    validate_part,
)


def test_text_ui_part():
    p = TextUIPart(text="hi", state="done")
    assert p.type == "text"
    assert p.model_dump(by_alias=True, exclude_none=True) == {
        "type": "text",
        "text": "hi",
        "state": "done",
    }


def test_file_ui_part_alias():
    p = FileUIPart.model_validate(
        {"type": "file", "mediaType": "image/png", "url": "data:image/png;base64,xx"}
    )
    assert p.media_type == "image/png"
    assert p.model_dump(by_alias=True, exclude_none=True) == {
        "type": "file",
        "mediaType": "image/png",
        "url": "data:image/png;base64,xx",
    }


def test_data_part_must_have_data_prefix():
    DataUIPart.model_validate({"type": "data-progress", "data": {"pct": 50}})
    with pytest.raises(ValueError):
        DataUIPart.model_validate({"type": "progress", "data": {}})


def test_tool_part_validates_prefix():
    ToolUIPart.model_validate(
        {
            "type": "tool-search",
            "toolCallId": "c",
            "state": "input-streaming",
            "input": {"q": "x"},
        }
    )
    with pytest.raises(ValueError):
        ToolUIPart.model_validate(
            {"type": "search", "toolCallId": "c", "state": "input-streaming"}
        )


def test_dynamic_tool_part():
    p = DynamicToolUIPart.model_validate(
        {
            "type": "dynamic-tool",
            "toolCallId": "c",
            "toolName": "delete_file",
            "state": "output-available",
            "input": {"f": "x.pdf"},
            "output": "ok",
        }
    )
    assert p.tool_name == "delete_file"
    assert p.state == "output-available"


def test_validate_part_dispatch():
    assert isinstance(validate_part({"type": "text", "text": "hi"}), TextUIPart)
    assert isinstance(
        validate_part({"type": "reasoning", "text": "x"}), ReasoningUIPart
    )
    assert isinstance(validate_part({"type": "step-start"}), StepStartUIPart)
    assert isinstance(
        validate_part(
            {
                "type": "tool-foo",
                "toolCallId": "c",
                "state": "input-streaming",
                "input": {},
            }
        ),
        ToolUIPart,
    )
    assert isinstance(
        validate_part({"type": "data-bar", "data": 1}),
        DataUIPart,
    )
    with pytest.raises(ValueError):
        validate_part({"type": "unknown"})


def test_guards():
    assert is_text_ui_part({"type": "text"})
    assert is_reasoning_ui_part({"type": "reasoning"})
    assert is_file_ui_part({"type": "file"})
    assert is_step_start_ui_part({"type": "step-start"})
    assert is_data_ui_part({"type": "data-foo"})
    assert is_static_tool_ui_part({"type": "tool-foo"})
    assert is_dynamic_tool_ui_part({"type": "dynamic-tool"})
    assert is_tool_ui_part({"type": "tool-foo"})
    assert is_tool_ui_part({"type": "dynamic-tool"})
    assert not is_tool_ui_part({"type": "text"})
    assert not is_data_ui_part({"type": "text"})


def test_get_tool_name():
    assert get_tool_name({"type": "tool-search"}) == "search"
    assert get_tool_name({"type": "tool-foo-bar"}) == "foo-bar"
    assert get_tool_name({"type": "dynamic-tool", "toolName": "x"}) == "x"
    assert get_static_tool_name({"type": "tool-search"}) == "search"


def test_uimessage_minimal():
    UIMessage(role="user", parts=[{"type": "text", "text": "hi"}])
