"""Process LangGraph stream events from `graph.astream(stream_mode=['values','messages'])`.

Mirrors `/tmp/ai-sdk-repo/packages/langchain/src/utils.ts:993-1654`
(`processLangGraphEvent` and `parseLangGraphEvent`).

The state machine handles three event kinds:
  - 'custom': user-emitted via writer() → emitted as `data-{name}` chunks
  - 'messages': streaming AIMessageChunk / ToolMessage from the graph
  - 'values': accumulated graph state with `messages` and optional `__interrupt__`
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from .extractors import (
    extract_image_outputs,
    extract_reasoning_from_content_blocks,
    extract_reasoning_from_values_message,
    extract_reasoning_id,
    get_message_id,
    get_message_text,
)
from .guards import (
    is_ai_message_chunk,
    is_plain_message_object,
    is_tool_message_type,
)
from .types import LangGraphEventState, MessageSeenEntry, ToolCallInfo

Emit = Callable[[dict[str, Any]], None]


def _now_ms() -> int:
    """Test seam — patch this to mock `Date.now()` in HITL ID generation."""
    return int(time.time() * 1000)


def parse_langgraph_event(event: list | tuple) -> tuple[Any, Any]:
    """utils.ts:34-38 — `[ns, type, data]` or `[type, data]`."""
    if len(event) == 3:
        return event[1], event[2]
    return event[0], event[1]


def _kwargs_or_self(msg: Any) -> dict[str, Any]:
    """For serialized messages, return `kwargs`; for plain dicts return msg; for instances build a synthetic dict."""
    if isinstance(msg, dict):
        if msg.get("type") == "constructor" and isinstance(msg.get("kwargs"), dict):
            return msg["kwargs"]
        return msg
    return {
        "id": getattr(msg, "id", None),
        "additional_kwargs": getattr(msg, "additional_kwargs", None),
        "tool_call_chunks": getattr(msg, "tool_call_chunks", None),
        "tool_calls": getattr(msg, "tool_calls", None),
        "content": getattr(msg, "content", None),
        "tool_call_id": getattr(msg, "tool_call_id", None),
        "status": getattr(msg, "status", None),
    }


def _ensure_seen(state: LangGraphEventState, msg_id: str) -> MessageSeenEntry:
    if msg_id not in state.message_seen:
        state.message_seen[msg_id] = MessageSeenEntry()
    return state.message_seen[msg_id]


def process_langgraph_event(
    event: list | tuple,
    state: LangGraphEventState,
    emit: Emit,
) -> None:
    """utils.ts:1000-1654."""
    type_, data = parse_langgraph_event(event)

    if type_ == "custom":
        _handle_custom(data, emit)
        return

    if type_ == "messages":
        _handle_messages(data, state, emit)
        return

    if type_ == "values":
        _handle_values(data, state, emit)
        return


# ---------- 'custom' ----------


def _handle_custom(data: Any, emit: Emit) -> None:
    """utils.ts:1018-1050."""
    custom_type_name = "custom"
    part_id: str | None = None
    if isinstance(data, dict):
        t = data.get("type")
        if isinstance(t, str) and t:
            custom_type_name = t
        i = data.get("id")
        if isinstance(i, str) and i:
            part_id = i
    chunk: dict[str, Any] = {
        "type": f"data-{custom_type_name}",
        "transient": part_id is None,
        "data": data,
    }
    if part_id is not None:
        chunk["id"] = part_id
    emit(chunk)


# ---------- 'messages' ----------


def _handle_messages(data: Any, state: LangGraphEventState, emit: Emit) -> None:
    """utils.ts:1052-1326."""
    if not isinstance(data, (list, tuple)) or len(data) < 1:
        return
    msg = data[0] if len(data) >= 1 else None
    metadata = data[1] if len(data) >= 2 else None
    if msg is None:
        return

    msg_id = get_message_id(msg)
    if not msg_id:
        return

    # Step boundary tracking (utils.ts:1063-1090)
    langgraph_step = (
        metadata.get("langgraph_step")
        if isinstance(metadata, dict)
        else None
    )
    if isinstance(langgraph_step, int) and langgraph_step != state.current_step:
        if state.current_step is not None:
            for id_, seen in list(state.message_seen.items()):
                if seen.text:
                    emit({"type": "text-end", "id": id_})
                if seen.reasoning:
                    emit({"type": "reasoning-end", "id": id_})
                state.message_seen.pop(id_, None)
                state.message_concat.pop(id_, None)
                state.message_reasoning_ids.pop(id_, None)
            emit({"type": "finish-step"})
        emit({"type": "start-step"})
        state.current_step = langgraph_step

    if is_ai_message_chunk(msg):
        _handle_ai_message_chunk(msg, msg_id, state, emit)
    elif is_tool_message_type(msg):
        _handle_tool_message(msg, state, emit)


def _handle_ai_message_chunk(
    msg: Any, msg_id: str, state: LangGraphEventState, emit: Emit
) -> None:
    """utils.ts:1106-1290."""
    src = _kwargs_or_self(msg)

    # Image generation outputs
    additional_kwargs = src.get("additional_kwargs")
    image_outputs = extract_image_outputs(
        additional_kwargs if isinstance(additional_kwargs, dict) else None
    )
    for image_output in image_outputs:
        image_id = image_output.get("id")
        if not image_id or image_id in state.emitted_images:
            continue
        if not image_output.get("result"):
            continue
        state.emitted_images.add(image_id)
        media_type = f"image/{image_output.get('output_format', 'png')}"
        emit(
            {
                "type": "file",
                "mediaType": media_type,
                "url": f"data:{media_type};base64,{image_output['result']}",
            }
        )

    # Streaming tool call chunks (utils.ts:1155-1228)
    tool_call_chunks = src.get("tool_call_chunks")
    if isinstance(tool_call_chunks, list) and tool_call_chunks:
        for tc in tool_call_chunks:
            if not isinstance(tc, dict):
                continue
            idx = tc.get("index", 0)

            if tc.get("id"):
                state.tool_call_info_by_index.setdefault(msg_id, {})
                state.tool_call_info_by_index[msg_id][idx] = ToolCallInfo(
                    id=tc["id"],
                    name=tc.get("name") or "unknown",
                )

            tool_call_id = tc.get("id") or (
                state.tool_call_info_by_index.get(msg_id, {}).get(idx).id
                if state.tool_call_info_by_index.get(msg_id, {}).get(idx)
                else None
            )
            if not tool_call_id:
                continue

            tool_name = (
                tc.get("name")
                or (
                    state.tool_call_info_by_index.get(msg_id, {}).get(idx).name
                    if state.tool_call_info_by_index.get(msg_id, {}).get(idx)
                    else None
                )
                or "unknown"
            )

            seen = _ensure_seen(state, msg_id)
            if not seen.tool.get(tool_call_id):
                emit(
                    {
                        "type": "tool-input-start",
                        "toolCallId": tool_call_id,
                        "toolName": tool_name,
                        "dynamic": True,
                    }
                )
                seen.tool[tool_call_id] = True
                state.emitted_tool_calls.add(tool_call_id)

            if tc.get("args"):
                emit(
                    {
                        "type": "tool-input-delta",
                        "toolCallId": tool_call_id,
                        "inputTextDelta": tc["args"],
                    }
                )
        return  # utils.ts:1228 — return; tool chunks short-circuit

    # Reasoning lifecycle (utils.ts:1241-1272)
    chunk_reasoning_id = extract_reasoning_id(msg)
    if chunk_reasoning_id:
        if msg_id not in state.message_reasoning_ids:
            state.message_reasoning_ids[msg_id] = chunk_reasoning_id
        state.emitted_reasoning_ids.add(chunk_reasoning_id)

    reasoning = extract_reasoning_from_content_blocks(msg)
    if reasoning:
        seen = _ensure_seen(state, msg_id)
        reasoning_id = (
            state.message_reasoning_ids.get(msg_id) or chunk_reasoning_id or msg_id
        )
        if not seen.reasoning:
            emit({"type": "reasoning-start", "id": msg_id})
            seen.reasoning = True
        emit({"type": "reasoning-delta", "delta": reasoning, "id": msg_id})
        state.emitted_reasoning_ids.add(reasoning_id)

    # Text content (utils.ts:1277-1290)
    text = get_message_text(msg)
    if text:
        seen = _ensure_seen(state, msg_id)
        if not seen.text:
            emit({"type": "text-start", "id": msg_id})
            seen.text = True
        emit({"type": "text-delta", "delta": text, "id": msg_id})


def _handle_tool_message(msg: Any, state: LangGraphEventState, emit: Emit) -> None:
    """utils.ts:1291-1324."""
    src = _kwargs_or_self(msg)
    tool_call_id = src.get("tool_call_id")
    status = src.get("status")
    if not tool_call_id:
        return

    if status == "error":
        content = src.get("content")
        emit(
            {
                "type": "tool-output-error",
                "toolCallId": tool_call_id,
                "errorText": content if isinstance(content, str) else "Tool execution failed",
            }
        )
    else:
        emit(
            {
                "type": "tool-output-available",
                "toolCallId": tool_call_id,
                "output": src.get("content"),
            }
        )


# ---------- 'values' ----------


def _handle_values(data: Any, state: LangGraphEventState, emit: Emit) -> None:
    """utils.ts:1329-1652."""
    # 1) Finalize all pending message chunks (utils.ts:1333-1365)
    for id_, seen in list(state.message_seen.items()):
        if seen.text:
            emit({"type": "text-end", "id": id_})
        if seen.tool:
            concat_msg = state.message_concat.get(id_)
            for tool_call_id, was_seen in list(seen.tool.items()):
                if not was_seen or concat_msg is None:
                    continue
                tool_call = _find_tool_call(concat_msg, tool_call_id)
                if not tool_call:
                    continue
                state.emitted_tool_calls.add(tool_call_id)
                key = f"{tool_call.get('name', '')}:{json.dumps(tool_call.get('args'))}"
                state.emitted_tool_calls_by_key[key] = tool_call_id
                emit(
                    {
                        "type": "tool-input-available",
                        "toolCallId": tool_call_id,
                        "toolName": tool_call.get("name"),
                        "input": tool_call.get("args"),
                        "dynamic": True,
                    }
                )
        if seen.reasoning:
            emit({"type": "reasoning-end", "id": id_})
        state.message_seen.pop(id_, None)
        state.message_concat.pop(id_, None)
        state.message_reasoning_ids.pop(id_, None)

    # 2) Walk values.messages for unstreamed tool calls / reasoning (utils.ts:1370-1568)
    if isinstance(data, dict) and "messages" in data:
        messages = data.get("messages")
        if isinstance(messages, list):
            _emit_values_messages(messages, state, emit)

    # 3) HITL interrupts (utils.ts:1571-1649)
    if isinstance(data, dict):
        interrupt = data.get("__interrupt__")
        if isinstance(interrupt, list) and interrupt:
            _emit_hitl_interrupts(interrupt, state, emit)


def _find_tool_call(concat_msg: Any, tool_call_id: str) -> dict | None:
    """Look up a tool_calls entry by id within an accumulated AIMessageChunk-like."""
    src = _kwargs_or_self(concat_msg)
    tool_calls = src.get("tool_calls")
    if not isinstance(tool_calls, list):
        return None
    for tc in tool_calls:
        if isinstance(tc, dict) and tc.get("id") == tool_call_id:
            return tc
    return None


def _emit_values_messages(
    messages: list, state: LangGraphEventState, emit: Emit
) -> None:
    """utils.ts:1378-1567 — emit unstreamed tool calls + values-only reasoning."""
    completed_tool_call_ids: set[str] = set()
    for msg in messages:
        if not isinstance(msg, (dict, object)):
            continue
        if is_tool_message_type(msg):
            src = _kwargs_or_self(msg)
            tcid = src.get("tool_call_id")
            if isinstance(tcid, str):
                completed_tool_call_ids.add(tcid)

    for msg in messages:
        if msg is None:
            continue
        msg_id = get_message_id(msg)
        if not msg_id:
            continue

        tool_calls = _extract_value_tool_calls(msg)

        if tool_calls:
            for tool_call in tool_calls:
                tcid = tool_call.get("id")
                if not tcid:
                    continue
                if tcid in state.emitted_tool_calls:
                    continue
                if tcid in completed_tool_call_ids:
                    continue
                state.emitted_tool_calls.add(tcid)
                key = f"{tool_call.get('name', '')}:{json.dumps(tool_call.get('args'))}"
                state.emitted_tool_calls_by_key[key] = tcid
                emit(
                    {
                        "type": "tool-input-start",
                        "toolCallId": tcid,
                        "toolName": tool_call.get("name"),
                        "dynamic": True,
                    }
                )
                emit(
                    {
                        "type": "tool-input-available",
                        "toolCallId": tcid,
                        "toolName": tool_call.get("name"),
                        "input": tool_call.get("args"),
                        "dynamic": True,
                    }
                )

        # Reasoning emission decision (utils.ts:1532-1565)
        reasoning_id = extract_reasoning_id(msg)
        was_streamed_this_request = msg_id in state.message_seen
        has_tool_calls = bool(tool_calls)

        should_emit_reasoning = (
            reasoning_id is not None
            and reasoning_id not in state.emitted_reasoning_ids
            and (was_streamed_this_request or not has_tool_calls)
        )

        if should_emit_reasoning:
            reasoning = extract_reasoning_from_values_message(msg)
            if reasoning:
                emit({"type": "reasoning-start", "id": msg_id})
                emit({"type": "reasoning-delta", "delta": reasoning, "id": msg_id})
                emit({"type": "reasoning-end", "id": msg_id})
                state.emitted_reasoning_ids.add(reasoning_id)


def _extract_value_tool_calls(msg: Any) -> list[dict]:
    """utils.ts:1413-1481 — pull tool_calls from an AI message in various formats."""
    if msg is None:
        return []

    # Class instances
    tool_calls_attr = getattr(msg, "tool_calls", None)
    if isinstance(tool_calls_attr, list):
        return [tc for tc in tool_calls_attr if isinstance(tc, dict)]

    if not is_plain_message_object(msg):
        return []

    if not isinstance(msg, dict):
        return []

    is_serialized = msg.get("type") == "constructor" and isinstance(msg.get("id"), list) and (
        "AIMessageChunk" in msg["id"] or "AIMessage" in msg["id"]
    )
    is_ai_plain = msg.get("type") == "ai"

    if not (is_serialized or is_ai_plain):
        return []

    src = msg.get("kwargs") if is_serialized else msg
    if not isinstance(src, dict):
        return []

    tool_calls = src.get("tool_calls")
    if isinstance(tool_calls, list):
        return [tc for tc in tool_calls if isinstance(tc, dict)]

    # Fallback: additional_kwargs.tool_calls (OpenAI raw format)
    ak = src.get("additional_kwargs")
    if isinstance(ak, dict) and isinstance(ak.get("tool_calls"), list):
        result: list[dict] = []
        for idx, tc in enumerate(ak["tool_calls"]):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            args_str = fn.get("arguments")
            try:
                args = json.loads(args_str) if args_str else {}
            except Exception:
                args = {}
            result.append(
                {
                    "id": tc.get("id") or f"call_{idx}",
                    "name": fn.get("name") or "unknown",
                    "args": args,
                }
            )
        return result
    return []


def _emit_hitl_interrupts(
    interrupt: list, state: LangGraphEventState, emit: Emit
) -> None:
    """utils.ts:1577-1646."""
    for item in interrupt:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if not isinstance(value, dict):
            continue
        action_requests = value.get("actionRequests") or value.get("action_requests")
        if not isinstance(action_requests, list):
            continue
        for ar in action_requests:
            if not isinstance(ar, dict):
                continue
            tool_name = ar.get("name")
            input_value = ar.get("args") if "args" in ar else ar.get("arguments")

            key = f"{tool_name}:{json.dumps(input_value)}"
            tool_call_id = (
                state.emitted_tool_calls_by_key.get(key)
                or ar.get("id")
                or f"hitl-{tool_name}-{_now_ms()}"
            )

            if tool_call_id not in state.emitted_tool_calls:
                state.emitted_tool_calls.add(tool_call_id)
                state.emitted_tool_calls_by_key[key] = tool_call_id
                emit(
                    {
                        "type": "tool-input-start",
                        "toolCallId": tool_call_id,
                        "toolName": tool_name,
                        "dynamic": True,
                    }
                )
                emit(
                    {
                        "type": "tool-input-available",
                        "toolCallId": tool_call_id,
                        "toolName": tool_name,
                        "input": input_value,
                        "dynamic": True,
                    }
                )

            emit(
                {
                    "type": "tool-approval-request",
                    "approvalId": tool_call_id,
                    "toolCallId": tool_call_id,
                }
            )
