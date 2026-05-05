"""Convert UIMessages from `useChat` into ModelMessages for downstream models.

Full port of `/tmp/ai-sdk-repo/packages/ai/src/ui/convert-to-model-messages.ts`.

The TypeScript signature is generic over `UI_MESSAGE`, `tools`, and `convertDataPart`;
the Python port keeps the same behavior with looser typing — a converted dict shape
is what downstream consumers operate on (LangChain conversion, providers, etc.).
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from .model_messages import MessageConversionError, ModelMessage, create_tool_model_output
from .ui_messages import (
    UIMessage,
    get_tool_name,
    is_custom_content_ui_part,
    is_data_ui_part,
    is_file_ui_part,
    is_reasoning_file_ui_part,
    is_reasoning_ui_part,
    is_text_ui_part,
    is_tool_ui_part,
)

ConvertDataPart = Callable[[Any], Any | None]
"""Optional converter: data part → TextPart-shaped or FilePart-shaped dict, or None to drop."""

ToolSet = dict[str, Any]


def _f(part: Any, py_name: str, alias: str | None = None) -> Any:
    """Read a field from a dict or Pydantic model with both snake_case and camelCase fallbacks."""
    if isinstance(part, dict):
        if alias and alias in part:
            return part[alias]
        return part.get(py_name)
    return getattr(part, py_name, None)


def _set(d: dict, key: str, value: Any) -> None:
    if value is not None:
        d[key] = value


async def convert_to_model_messages(
    messages: list[Any],
    *,
    tools: ToolSet | None = None,
    ignore_incomplete_tool_calls: bool = False,
    convert_data_part: ConvertDataPart | None = None,
) -> list[dict]:
    """Mirror ai/src/ui/convert-to-model-messages.ts.

    Returns a list of ModelMessage-shaped dicts (TS-compatible JSON, camelCase-aliased).
    Accepts UIMessage Pydantic models or raw dicts in `messages`.
    """
    # Normalize UIMessage Pydantic models → dicts so we can use string field access uniformly.
    norm: list[dict] = []
    for m in messages:
        if isinstance(m, UIMessage):
            norm.append(m.model_dump(by_alias=True, exclude_none=True))
        elif isinstance(m, dict):
            norm.append(m)
        else:
            norm.append(dict(m))  # type: ignore[arg-type]

    if ignore_incomplete_tool_calls:
        for msg in norm:
            msg["parts"] = [
                p
                for p in (msg.get("parts") or [])
                if not is_tool_ui_part(p)
                or _f(p, "state") not in ("input-streaming", "input-available")
            ]

    model_messages: list[dict] = []

    for message in norm:
        role = message.get("role")
        parts: list[Any] = list(message.get("parts") or [])

        if role == "system":
            text_parts = [p for p in parts if is_text_ui_part(p)]
            provider_metadata: dict[str, Any] = {}
            for p in text_parts:
                pm = _f(p, "provider_metadata", "providerMetadata")
                if pm:
                    provider_metadata.update(pm)
            entry: dict[str, Any] = {
                "role": "system",
                "content": "".join(_f(p, "text") or "" for p in text_parts),
            }
            if provider_metadata:
                entry["providerOptions"] = provider_metadata
            model_messages.append(entry)
            continue

        if role == "user":
            content: list[dict] = []
            for p in parts:
                converted = _convert_user_part(p, convert_data_part)
                if converted is not None:
                    content.append(converted)
            model_messages.append({"role": "user", "content": content})
            continue

        if role == "assistant":
            await _process_assistant(
                parts=parts,
                model_messages=model_messages,
                tools=tools,
                convert_data_part=convert_data_part,
            )
            continue

        raise MessageConversionError(
            original_message=message,
            message=f"Unsupported role: {role}",
        )

    return model_messages


def _convert_user_part(part: Any, convert_data_part: ConvertDataPart | None) -> dict | None:
    if is_text_ui_part(part):
        out: dict[str, Any] = {"type": "text", "text": _f(part, "text")}
        pm = _f(part, "provider_metadata", "providerMetadata")
        if pm is not None:
            out["providerOptions"] = pm
        return out
    if is_file_ui_part(part):
        ref = _f(part, "provider_reference", "providerReference")
        url = _f(part, "url")
        data = (
            {"type": "reference", "reference": ref}
            if ref is not None
            else {"type": "url", "url": url}
        )
        out = {
            "type": "file",
            "mediaType": _f(part, "media_type", "mediaType"),
            "data": data,
        }
        _set(out, "filename", _f(part, "filename"))
        pm = _f(part, "provider_metadata", "providerMetadata")
        if pm is not None:
            out["providerOptions"] = pm
        return out
    if is_data_ui_part(part) and convert_data_part is not None:
        return convert_data_part(part)
    return None


async def _process_assistant(
    *,
    parts: list[Any],
    model_messages: list[dict],
    tools: ToolSet | None,
    convert_data_part: ConvertDataPart | None,
) -> None:
    """Mirror the inner block-by-block processing for assistant messages.

    Blocks are delimited by `step-start` parts.
    """
    block: list[Any] = []

    async def process_block() -> None:
        if not block:
            return

        content: list[dict] = []
        for part in block:
            if is_text_ui_part(part):
                entry: dict[str, Any] = {"type": "text", "text": _f(part, "text")}
                pm = _f(part, "provider_metadata", "providerMetadata")
                if pm is not None:
                    entry["providerOptions"] = pm
                content.append(entry)
            elif is_custom_content_ui_part(part):
                entry = {"type": "custom", "kind": _f(part, "kind")}
                pm = _f(part, "provider_metadata", "providerMetadata")
                if pm is not None:
                    entry["providerOptions"] = pm
                content.append(entry)
            elif is_file_ui_part(part):
                ref = _f(part, "provider_reference", "providerReference")
                url = _f(part, "url")
                data = (
                    {"type": "reference", "reference": ref}
                    if ref is not None
                    else {"type": "url", "url": url}
                )
                entry = {
                    "type": "file",
                    "mediaType": _f(part, "media_type", "mediaType"),
                    "data": data,
                }
                _set(entry, "filename", _f(part, "filename"))
                pm = _f(part, "provider_metadata", "providerMetadata")
                if pm is not None:
                    entry["providerOptions"] = pm
                content.append(entry)
            elif is_reasoning_file_ui_part(part):
                content.append(
                    {
                        "type": "reasoning-file",
                        "data": {"type": "url", "url": _f(part, "url")},
                        "mediaType": _f(part, "media_type", "mediaType"),
                        "providerOptions": _f(part, "provider_metadata", "providerMetadata"),
                    }
                )
            elif is_reasoning_ui_part(part):
                content.append(
                    {
                        "type": "reasoning",
                        "text": _f(part, "text"),
                        "providerOptions": _f(part, "provider_metadata", "providerMetadata"),
                    }
                )
            elif is_tool_ui_part(part):
                tool_name = get_tool_name(part)
                state = _f(part, "state")
                if state != "input-streaming":
                    raw_input = _f(part, "raw_input", "rawInput")
                    input_value = _f(part, "input")
                    if state == "output-error":
                        input_value = (
                            input_value if input_value is not None else raw_input
                        )

                    call_meta = _f(part, "call_provider_metadata", "callProviderMetadata")
                    tc: dict[str, Any] = {
                        "type": "tool-call",
                        "toolCallId": _f(part, "tool_call_id", "toolCallId"),
                        "toolName": tool_name,
                        "input": input_value,
                        "providerExecuted": _f(part, "provider_executed", "providerExecuted"),
                    }
                    if call_meta is not None:
                        tc["providerOptions"] = call_meta
                    content.append(tc)

                    approval = _f(part, "approval")
                    if approval is not None:
                        approval_id = (
                            approval.get("id")
                            if isinstance(approval, dict)
                            else getattr(approval, "id", None)
                        )
                        is_auto = (
                            approval.get("isAutomatic")
                            if isinstance(approval, dict)
                            else getattr(approval, "is_automatic", None)
                        )
                        ar: dict[str, Any] = {
                            "type": "tool-approval-request",
                            "approvalId": approval_id,
                            "toolCallId": _f(part, "tool_call_id", "toolCallId"),
                        }
                        _set(ar, "isAutomatic", is_auto)
                        content.append(ar)

                    if (
                        _f(part, "provider_executed", "providerExecuted") is True
                        and state != "approval-responded"
                        and state in ("output-available", "output-error")
                    ):
                        result_meta = _f(
                            part, "result_provider_metadata", "resultProviderMetadata"
                        ) or call_meta
                        output_value = (
                            _f(part, "error_text", "errorText")
                            if state == "output-error"
                            else _f(part, "output")
                        )
                        result_entry: dict[str, Any] = {
                            "type": "tool-result",
                            "toolCallId": _f(part, "tool_call_id", "toolCallId"),
                            "toolName": tool_name,
                            "output": await create_tool_model_output(
                                tool_call_id=_f(part, "tool_call_id", "toolCallId"),
                                input=_f(part, "input"),
                                output=output_value,
                                tool=(tools or {}).get(tool_name) if tools else None,
                                error_mode="json" if state == "output-error" else "none",
                            ),
                        }
                        if result_meta is not None:
                            result_entry["providerOptions"] = result_meta
                        content.append(result_entry)
            elif is_data_ui_part(part):
                if convert_data_part is not None:
                    converted = convert_data_part(part)
                    if converted is not None:
                        content.append(converted)
            else:
                raise ValueError(f"Unsupported part: {part!r}")

        model_messages.append({"role": "assistant", "content": content})

        # Tool message follow-up: include non-provider-executed tools, OR provider-executed
        # tools with approval responses.
        tool_parts = [
            p
            for p in block
            if is_tool_ui_part(p)
            and (
                _f(p, "provider_executed", "providerExecuted") is not True
                or (
                    _f(p, "approval") is not None
                    and (
                        _f(p, "approval").get("approved")
                        if isinstance(_f(p, "approval"), dict)
                        else getattr(_f(p, "approval"), "approved", None)
                    )
                    is not None
                )
            )
        ]
        if tool_parts:
            tool_content: list[dict] = []
            for tp in tool_parts:
                approval = _f(tp, "approval")
                approval_approved = None
                approval_id = None
                approval_reason = None
                if approval is not None:
                    approval_approved = (
                        approval.get("approved")
                        if isinstance(approval, dict)
                        else getattr(approval, "approved", None)
                    )
                    approval_id = (
                        approval.get("id")
                        if isinstance(approval, dict)
                        else getattr(approval, "id", None)
                    )
                    approval_reason = (
                        approval.get("reason")
                        if isinstance(approval, dict)
                        else getattr(approval, "reason", None)
                    )

                if approval_approved is not None:
                    resp: dict[str, Any] = {
                        "type": "tool-approval-response",
                        "approvalId": approval_id,
                        "approved": approval_approved,
                    }
                    _set(resp, "reason", approval_reason)
                    _set(
                        resp,
                        "providerExecuted",
                        _f(tp, "provider_executed", "providerExecuted"),
                    )
                    tool_content.append(resp)

                state = _f(tp, "state")
                if state == "approval-responded" and approval_approved is False:
                    call_meta = _f(tp, "call_provider_metadata", "callProviderMetadata")
                    denied: dict[str, Any] = {
                        "type": "tool-result",
                        "toolCallId": _f(tp, "tool_call_id", "toolCallId"),
                        "toolName": get_tool_name(tp),
                        "output": {"type": "execution-denied", "reason": approval_reason},
                    }
                    if call_meta is not None:
                        denied["providerOptions"] = call_meta
                    tool_content.append(denied)

                if _f(tp, "provider_executed", "providerExecuted") is True:
                    continue

                if state == "output-denied":
                    call_meta = _f(tp, "call_provider_metadata", "callProviderMetadata")
                    err: dict[str, Any] = {
                        "type": "tool-result",
                        "toolCallId": _f(tp, "tool_call_id", "toolCallId"),
                        "toolName": get_tool_name(tp),
                        "output": {
                            "type": "error-text",
                            "value": approval_reason or "Tool call execution denied.",
                        },
                    }
                    if call_meta is not None:
                        err["providerOptions"] = call_meta
                    tool_content.append(err)
                elif state in ("output-error", "output-available"):
                    call_meta = _f(tp, "call_provider_metadata", "callProviderMetadata")
                    tool_name = get_tool_name(tp)
                    output_value = (
                        _f(tp, "error_text", "errorText")
                        if state == "output-error"
                        else _f(tp, "output")
                    )
                    res: dict[str, Any] = {
                        "type": "tool-result",
                        "toolCallId": _f(tp, "tool_call_id", "toolCallId"),
                        "toolName": tool_name,
                        "output": await create_tool_model_output(
                            tool_call_id=_f(tp, "tool_call_id", "toolCallId"),
                            input=_f(tp, "input"),
                            output=output_value,
                            tool=(tools or {}).get(tool_name) if tools else None,
                            error_mode="text" if state == "output-error" else "none",
                        ),
                    }
                    if call_meta is not None:
                        res["providerOptions"] = call_meta
                    tool_content.append(res)

            if tool_content:
                model_messages.append({"role": "tool", "content": tool_content})

        block.clear()

    for part in parts:
        t = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
        if (
            is_custom_content_ui_part(part)
            or is_text_ui_part(part)
            or is_reasoning_ui_part(part)
            or is_reasoning_file_ui_part(part)
            or is_file_ui_part(part)
            or is_tool_ui_part(part)
            or is_data_ui_part(part)
        ):
            block.append(part)
        elif t == "step-start":
            await process_block()

    await process_block()
