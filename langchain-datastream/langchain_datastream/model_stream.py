"""Process a single AIMessageChunk from `model.astream()`.

Mirrors `/tmp/ai-sdk-repo/packages/langchain/src/utils.ts:424-549` (`processModelChunk`).

The TS controller is a `ReadableStreamDefaultController<UIMessageChunk>`. In Python
we accept a callable `emit: Callable[[dict], None]` that pushes chunk dicts
(camelCase JSON shape) onto the consumer's queue/list/generator.
"""

from __future__ import annotations

from typing import Any, Callable

from .extractors import (
    extract_image_outputs,
    extract_reasoning_from_content_blocks,
    extract_reasoning_from_values_message,
    get_message_text,
)
from .types import ModelStreamState

Emit = Callable[[dict[str, Any]], None]


def _kwargs_or_attrs(chunk: Any) -> dict[str, Any]:
    """Get kwargs (serialized) or relevant attrs (class instance) as a dict."""
    if isinstance(chunk, dict):
        if chunk.get("type") == "constructor" and isinstance(chunk.get("kwargs"), dict):
            return chunk["kwargs"]
        return chunk
    return {
        "id": getattr(chunk, "id", None),
        "additional_kwargs": getattr(chunk, "additional_kwargs", None),
        "response_metadata": getattr(chunk, "response_metadata", None),
        "content": getattr(chunk, "content", None),
    }


def process_model_chunk(chunk: Any, state: ModelStreamState, emit: Emit) -> None:
    """utils.ts:424-549.

    Sequence of emits per chunk:
      1. Image generation outputs → `file` chunks (deduped via `state.emitted_images`)
      2. Reasoning lifecycle → `reasoning-start` / `reasoning-delta`
      3. Text lifecycle → `text-start` / `text-delta` (closes any open reasoning first)

    `*-end` events are emitted by the orchestrator at stream end, not here.
    """
    kw = _kwargs_or_attrs(chunk)

    chunk_id = kw.get("id")
    if isinstance(chunk_id, str):
        state.message_id = chunk_id

    # 1) Images via tool_outputs → file chunks
    additional_kwargs = kw.get("additional_kwargs")
    image_outputs = extract_image_outputs(
        additional_kwargs if isinstance(additional_kwargs, dict) else None
    )
    for image_output in image_outputs:
        image_id = image_output.get("id")
        if not isinstance(image_id, str) or image_id in state.emitted_images:
            continue
        state.emitted_images.add(image_id)
        media_type = f"image/{image_output.get('output_format', 'png')}"
        emit(
            {
                "type": "file",
                "mediaType": media_type,
                "url": f"data:{media_type};base64,{image_output.get('result', '')}",
            }
        )
        state.started = True

    # 2) Reasoning — for direct model streams we check both sources (utils.ts:486-503)
    reasoning = extract_reasoning_from_content_blocks(
        chunk
    ) or extract_reasoning_from_values_message(chunk)
    if reasoning:
        if not state.reasoning_started:
            state.reasoning_message_id = state.message_id
            emit({"type": "reasoning-start", "id": state.message_id})
            state.reasoning_started = True
            state.started = True
        emit(
            {
                "type": "reasoning-delta",
                "delta": reasoning,
                "id": state.reasoning_message_id or state.message_id,
            }
        )

    # 3) Text content
    text = get_message_text(chunk)
    if text:
        # Close reasoning before starting text (utils.ts:528-534)
        if state.reasoning_started and not state.text_started:
            emit(
                {
                    "type": "reasoning-end",
                    "id": state.reasoning_message_id or state.message_id,
                }
            )
            state.reasoning_started = False

        if not state.text_started:
            state.text_message_id = state.message_id
            emit({"type": "text-start", "id": state.message_id})
            state.text_started = True
            state.started = True

        emit(
            {
                "type": "text-delta",
                "delta": text,
                "id": state.text_message_id or state.message_id,
            }
        )
