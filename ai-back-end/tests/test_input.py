"""Phase 2d: ModelMessage → BaseMessage conversion."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ai_sdk.input import (
    convert_assistant_content,
    convert_model_messages,
    convert_tool_result_part,
    convert_user_content,
    to_base_messages,
)


def test_convert_user_content_string():
    msg = convert_user_content("hello")
    assert isinstance(msg, HumanMessage)
    assert msg.content == "hello"


def test_convert_user_content_text_only_collapses_to_string():
    msg = convert_user_content(
        [{"type": "text", "text": "hi"}, {"type": "text", "text": " there"}]
    )
    assert msg.content == "hi there"


def test_convert_user_content_with_image_url():
    msg = convert_user_content(
        [
            {"type": "text", "text": "look"},
            {
                "type": "file",
                "mediaType": "image/png",
                "data": {"type": "url", "url": "https://example.com/x.png"},
            },
        ]
    )
    assert msg.content == [
        {"type": "text", "text": "look"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/x.png"},
        },
    ]


def test_convert_user_content_with_image_base64():
    msg = convert_user_content(
        [
            {
                "type": "image",
                "image": "AAAA",
                "mediaType": "image/jpeg",
            }
        ]
    )
    assert msg.content == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,AAAA"},
        }
    ]


def test_convert_user_content_with_pdf_url():
    msg = convert_user_content(
        [
            {
                "type": "file",
                "mediaType": "application/pdf",
                "data": {"type": "url", "url": "https://example.com/a.pdf"},
            }
        ]
    )
    assert msg.content == [
        {
            "type": "file",
            "url": "https://example.com/a.pdf",
            "mimeType": "application/pdf",
            "filename": "file.pdf",
        }
    ]


def test_convert_assistant_content_string():
    msg = convert_assistant_content("done")
    assert isinstance(msg, AIMessage)
    assert msg.content == "done"
    assert msg.tool_calls == []


def test_convert_assistant_content_with_tool_call():
    msg = convert_assistant_content(
        [
            {"type": "text", "text": "calling..."},
            {
                "type": "tool-call",
                "toolCallId": "c1",
                "toolName": "search",
                "input": {"q": "x"},
            },
        ]
    )
    assert msg.content == "calling..."
    # LangChain normalizes tool_calls with a `type: 'tool_call'` field.
    assert msg.tool_calls == [
        {"id": "c1", "name": "search", "args": {"q": "x"}, "type": "tool_call"}
    ]


def test_convert_tool_result_part_text():
    msg = convert_tool_result_part(
        {
            "type": "tool-result",
            "toolCallId": "c1",
            "toolName": "search",
            "output": {"type": "text", "value": "found"},
        }
    )
    assert isinstance(msg, ToolMessage)
    assert msg.content == "found"
    assert msg.tool_call_id == "c1"


def test_convert_tool_result_part_json():
    msg = convert_tool_result_part(
        {
            "type": "tool-result",
            "toolCallId": "c1",
            "toolName": "search",
            "output": {"type": "json", "value": {"k": "v"}},
        }
    )
    assert msg.content == '{"k": "v"}'


def test_convert_tool_result_part_content():
    msg = convert_tool_result_part(
        {
            "type": "tool-result",
            "toolCallId": "c1",
            "toolName": "search",
            "output": {
                "type": "content",
                "value": [
                    {"type": "text", "text": "a"},
                    {"type": "image", "image": "..."},
                    {"type": "text", "text": "b"},
                ],
            },
        }
    )
    assert msg.content == "ab"


def test_convert_model_messages_full_round():
    msgs = convert_model_messages(
        [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "calling"},
                    {
                        "type": "tool-call",
                        "toolCallId": "c1",
                        "toolName": "x",
                        "input": {},
                    },
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool-result",
                        "toolCallId": "c1",
                        "toolName": "x",
                        "output": {"type": "text", "value": "ok"},
                    }
                ],
            },
        ]
    )
    assert isinstance(msgs[0], SystemMessage)
    assert isinstance(msgs[1], HumanMessage)
    assert isinstance(msgs[2], AIMessage)
    assert isinstance(msgs[3], ToolMessage)
    assert msgs[3].content == "ok"


@pytest.mark.asyncio
async def test_to_base_messages_end_to_end():
    msgs = await to_base_messages(
        [
            {"role": "user", "parts": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "parts": [{"type": "text", "text": "hi back"}]},
        ]
    )
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "hello"
    assert isinstance(msgs[1], AIMessage)
    assert msgs[1].content == "hi back"
