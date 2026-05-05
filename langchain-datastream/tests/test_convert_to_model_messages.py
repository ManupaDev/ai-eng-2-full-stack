"""Phase 2c parity: convert_to_model_messages mirrors the TS test suite.

Each test mirrors a scenario from
/tmp/ai-sdk-repo/packages/ai/src/ui/convert-to-model-messages.test.ts.

We compare against expected dict shapes rather than the raw .snap file because
Vitest's snapshot serializer renders `undefined` keys, while JSON.stringify
drops them — our Python output matches the JSON wire format.
"""

import pytest

from langchain_datastream.convert_to_model_messages import convert_to_model_messages
from langchain_datastream.model_messages import MessageConversionError

pytestmark = pytest.mark.asyncio


# ---------- system ----------


async def test_simple_system():
    result = await convert_to_model_messages(
        [{"role": "system", "parts": [{"text": "System message", "type": "text"}]}]
    )
    assert result == [{"role": "system", "content": "System message"}]


async def test_system_with_provider_metadata():
    result = await convert_to_model_messages(
        [
            {
                "role": "system",
                "parts": [
                    {
                        "text": "System message with metadata",
                        "type": "text",
                        "providerMetadata": {
                            "testProvider": {"systemSignature": "abc123"}
                        },
                    }
                ],
            }
        ]
    )
    assert result == [
        {
            "role": "system",
            "content": "System message with metadata",
            "providerOptions": {"testProvider": {"systemSignature": "abc123"}},
        }
    ]


async def test_system_merges_provider_metadata_from_multiple_text_parts():
    result = await convert_to_model_messages(
        [
            {
                "role": "system",
                "parts": [
                    {
                        "text": "Part 1",
                        "type": "text",
                        "providerMetadata": {"provider1": {"key1": "value1"}},
                    },
                    {
                        "text": " Part 2",
                        "type": "text",
                        "providerMetadata": {"provider2": {"key2": "value2"}},
                    },
                ],
            }
        ]
    )
    assert result == [
        {
            "role": "system",
            "content": "Part 1 Part 2",
            "providerOptions": {
                "provider1": {"key1": "value1"},
                "provider2": {"key2": "value2"},
            },
        }
    ]


# ---------- user ----------


async def test_simple_user():
    result = await convert_to_model_messages(
        [{"role": "user", "parts": [{"text": "Hello, AI!", "type": "text"}]}]
    )
    assert result == [
        {"role": "user", "content": [{"type": "text", "text": "Hello, AI!"}]}
    ]


async def test_user_with_provider_metadata():
    result = await convert_to_model_messages(
        [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "Hello, AI!",
                        "type": "text",
                        "providerMetadata": {
                            "testProvider": {"signature": "1234567890"}
                        },
                    }
                ],
            }
        ]
    )
    assert result == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hello, AI!",
                    "providerOptions": {
                        "testProvider": {"signature": "1234567890"}
                    },
                }
            ],
        }
    ]


async def test_user_with_file_parts():
    result = await convert_to_model_messages(
        [
            {
                "role": "user",
                "parts": [
                    {
                        "type": "file",
                        "mediaType": "image/jpeg",
                        "url": "https://example.com/image.jpg",
                    },
                    {"type": "text", "text": "Check this image"},
                ],
            }
        ]
    )
    assert result == [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "mediaType": "image/jpeg",
                    "data": {"type": "url", "url": "https://example.com/image.jpg"},
                },
                {"type": "text", "text": "Check this image"},
            ],
        }
    ]


async def test_user_file_with_provider_reference():
    result = await convert_to_model_messages(
        [
            {
                "role": "user",
                "parts": [
                    {
                        "type": "file",
                        "mediaType": "image/jpeg",
                        "url": "https://example.com/image.jpg",
                        "providerReference": {"openai": "file_abc"},
                    },
                ],
            }
        ]
    )
    assert result == [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "mediaType": "image/jpeg",
                    "data": {"type": "reference", "reference": {"openai": "file_abc"}},
                },
            ],
        }
    ]


# ---------- assistant ----------


async def test_simple_assistant_text():
    result = await convert_to_model_messages(
        [{"role": "assistant", "parts": [{"type": "text", "text": "Hi back"}]}]
    )
    assert result == [
        {"role": "assistant", "content": [{"type": "text", "text": "Hi back"}]}
    ]


async def test_assistant_with_reasoning():
    result = await convert_to_model_messages(
        [
            {
                "role": "assistant",
                "parts": [
                    {"type": "reasoning", "text": "thinking..."},
                    {"type": "text", "text": "answer"},
                ],
            }
        ]
    )
    assert result == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "reasoning",
                    "text": "thinking...",
                    "providerOptions": None,
                },
                {"type": "text", "text": "answer"},
            ],
        }
    ]


async def test_assistant_with_tool_output_available():
    result = await convert_to_model_messages(
        [
            {
                "role": "assistant",
                "parts": [
                    {"type": "text", "text": "i am gonna use tool1"},
                    {
                        "type": "tool-screenshot",
                        "toolCallId": "call-1",
                        "state": "output-available",
                        "input": {"value": "value-1"},
                        "output": "result-1",
                    },
                ],
            }
        ]
    )
    assert result == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "i am gonna use tool1"},
                {
                    "type": "tool-call",
                    "toolCallId": "call-1",
                    "toolName": "screenshot",
                    "input": {"value": "value-1"},
                    "providerExecuted": None,
                },
            ],
        },
        {
            "role": "tool",
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": "call-1",
                    "toolName": "screenshot",
                    "output": {"type": "text", "value": "result-1"},
                }
            ],
        },
    ]


async def test_assistant_step_start_splits_blocks():
    """step-start delimits assistant blocks; each block produces its own assistant + tool messages."""
    result = await convert_to_model_messages(
        [
            {
                "role": "assistant",
                "parts": [
                    {"type": "text", "text": "i am gonna use tool2 and tool3"},
                    {
                        "type": "tool-screenshot",
                        "toolCallId": "call-2",
                        "state": "output-available",
                        "input": {"value": "value-2"},
                        "output": "result-2",
                    },
                    {"type": "step-start"},
                    {"type": "text", "text": "all done"},
                ],
            }
        ]
    )
    # First block has text + tool-call → assistant; tool-result → tool. Then split.
    # Second block has text only → assistant.
    assert len(result) == 3
    assert result[0]["role"] == "assistant"
    assert result[1]["role"] == "tool"
    assert result[2] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "all done"}],
    }


async def test_unsupported_role_raises():
    with pytest.raises(MessageConversionError):
        await convert_to_model_messages([{"role": "alien", "parts": []}])


async def test_ignore_incomplete_tool_calls():
    result = await convert_to_model_messages(
        [
            {
                "role": "assistant",
                "parts": [
                    {"type": "text", "text": "starting"},
                    {
                        "type": "tool-foo",
                        "toolCallId": "c1",
                        "state": "input-streaming",
                        "input": {"partial": True},
                    },
                ],
            }
        ],
        ignore_incomplete_tool_calls=True,
    )
    assert result == [
        {"role": "assistant", "content": [{"type": "text", "text": "starting"}]}
    ]
