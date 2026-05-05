"""Process events from `agent.astream_events(version='v2')`.

Mirrors `/tmp/ai-sdk-repo/packages/langchain/src/adapter.ts:121-285` (`processStreamEventsEvent`).
"""

from __future__ import annotations

from typing import Any, Callable

from .extractors import extract_reasoning_from_content_blocks
from .types import ModelStreamState

Emit = Callable[[dict[str, Any]], None]


def process_stream_events_event(
    event: dict[str, Any],
    state: ModelStreamState,
    emit: Emit,
) -> None:
    """Handle one event from `astream_events`.

    Event shape: ``{event: str, data: dict | None, run_id?: str, name?: str}``.
    """
    run_id = event.get("run_id")
    if run_id and not state.started:
        state.message_id = run_id

    data = event.get("data")
    if not data:
        return

    event_name = event.get("event")

    if event_name == "on_chat_model_start":
        candidate = run_id or data.get("run_id") if isinstance(data, dict) else None
        if isinstance(candidate, str):
            state.message_id = candidate
        return

    if event_name == "on_chat_model_stream":
        chunk = data.get("chunk") if isinstance(data, dict) else None
        if not (chunk and isinstance(chunk, (dict, object))):
            return

        chunk_id = (
            chunk.get("id")
            if isinstance(chunk, dict)
            else getattr(chunk, "id", None)
        )
        if isinstance(chunk_id, str):
            state.message_id = chunk_id

        # Reasoning lifecycle (streaming-only — adapter.ts:182-197)
        reasoning = extract_reasoning_from_content_blocks(chunk)
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

        # Text content extraction (adapter.ts:201-217)
        content = (
            chunk.get("content")
            if isinstance(chunk, dict)
            else getattr(chunk, "content", None)
        )
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                c.get("text", "")
                for c in content
                if isinstance(c, dict)
                and c.get("type") == "text"
                and isinstance(c.get("text"), str)
            )

        if text:
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
        return

    if event_name == "on_tool_start":
        tool_run_id = run_id or (
            data.get("run_id") if isinstance(data, dict) else None
        )
        name = event.get("name") or (
            data.get("name") if isinstance(data, dict) else None
        )
        if tool_run_id and name:
            emit(
                {
                    "type": "tool-input-start",
                    "toolCallId": tool_run_id,
                    "toolName": name,
                    "dynamic": True,
                }
            )
        return

    if event_name == "on_tool_end":
        tool_run_id = run_id or (
            data.get("run_id") if isinstance(data, dict) else None
        )
        output = data.get("output") if isinstance(data, dict) else None
        if tool_run_id:
            emit(
                {
                    "type": "tool-output-available",
                    "toolCallId": tool_run_id,
                    "output": output,
                }
            )
        return
