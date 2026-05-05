"""Phase 4a verification: process_model_chunk."""

from langchain_core.messages import AIMessageChunk

from langchain_datastream.model_stream import process_model_chunk
from langchain_datastream.types import ModelStreamState


def _emit_to_list(out: list):
    return out.append


def test_text_only_chunk_starts_and_deltas():
    state = ModelStreamState()
    out: list = []
    process_model_chunk(
        AIMessageChunk(content="hello", id="run-1"), state, _emit_to_list(out)
    )
    assert out == [
        {"type": "text-start", "id": "run-1"},
        {"type": "text-delta", "delta": "hello", "id": "run-1"},
    ]
    assert state.text_started
    assert state.message_id == "run-1"


def test_subsequent_text_chunk_only_emits_delta():
    state = ModelStreamState()
    out: list = []
    process_model_chunk(
        AIMessageChunk(content="hi", id="run-1"), state, _emit_to_list(out)
    )
    out.clear()
    process_model_chunk(
        AIMessageChunk(content=" there", id="run-1"), state, _emit_to_list(out)
    )
    assert out == [{"type": "text-delta", "delta": " there", "id": "run-1"}]


def test_reasoning_then_text_closes_reasoning():
    state = ModelStreamState()
    out: list = []
    # Streaming reasoning chunk
    process_model_chunk(
        {
            "id": "r-1",
            "content": "",
            "additional_kwargs": {
                "reasoning": {"summary": [{"type": "summary_text", "text": "thinking"}]}
            },
        },
        state,
        _emit_to_list(out),
    )
    assert out == [
        {"type": "reasoning-start", "id": "r-1"},
        {"type": "reasoning-delta", "delta": "thinking", "id": "r-1"},
    ]

    out.clear()
    # Now a text chunk
    process_model_chunk({"id": "r-1", "content": "answer"}, state, _emit_to_list(out))
    assert out == [
        {"type": "reasoning-end", "id": "r-1"},
        {"type": "text-start", "id": "r-1"},
        {"type": "text-delta", "delta": "answer", "id": "r-1"},
    ]


def test_image_output_emits_file_chunk_and_dedupes():
    state = ModelStreamState()
    out: list = []
    chunk = {
        "id": "msg-1",
        "content": "",
        "additional_kwargs": {
            "tool_outputs": [
                {
                    "type": "image_generation_call",
                    "id": "img-1",
                    "result": "BASE64DATA",
                    "output_format": "png",
                }
            ]
        },
    }
    process_model_chunk(chunk, state, _emit_to_list(out))
    assert out == [
        {
            "type": "file",
            "mediaType": "image/png",
            "url": "data:image/png;base64,BASE64DATA",
        }
    ]

    # Same image again — deduped.
    out.clear()
    process_model_chunk(chunk, state, _emit_to_list(out))
    assert out == []


def test_array_content_text_blocks_join():
    state = ModelStreamState()
    out: list = []
    process_model_chunk(
        {
            "id": "m1",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "image", "url": "x"},
                {"type": "text", "text": "b"},
            ],
        },
        state,
        _emit_to_list(out),
    )
    assert out == [
        {"type": "text-start", "id": "m1"},
        {"type": "text-delta", "delta": "ab", "id": "m1"},
    ]
