# langchain-datastream

A Python port of [`@ai-sdk/langchain`](https://github.com/vercel/ai/tree/main/packages/langchain) — converts streams from LangChain Runnables and LangGraph graphs into the Vercel AI SDK **UI Message Stream** protocol consumed by `useChat`.

## Why

The TS package gives JS developers a one-liner to wire a LangGraph backend to a React `useChat` frontend. This package gives Python developers the same one-liner.

```python
from langchain_datastream import ui_message_stream_response, to_base_messages

@app.post("/api/chat")
async def chat(request: ChatRequest):
    messages = to_base_messages(request.messages)
    return ui_message_stream_response(
        agent.astream_events({"messages": messages}, version="v2"),
    )
```

## Stream sources supported

| Source | Function |
|---|---|
| LangChain `astream_events(version="v2")` (recommended for token streaming) | `to_ui_message_stream` |
| LangGraph `astream(stream_mode=["messages", "values", "updates"])` | `to_ui_message_stream` |
| Plain `BaseMessageChunk` async iterators | `to_ui_message_stream` |

## TS-parity notes

This package is a faithful port of the TS adapter, with two intentional Python-specific behaviors documented in `stream_events.py` and `sse.py`:

1. `on_tool_start` emits both `tool-input-start` *and* `tool-input-available` (TS only emits the former). This gives the UI a proper `Pending → Running → Completed` badge progression instead of `Pending → Completed`.
2. Tool outputs are coerced via `_coerce_tool_output` and the SSE encoder has a `default=` fallback. JS's `JSON.stringify` is permissive; Python's `json.dumps` is strict — without coercion, a `ToolMessage` from `ToolNode` would crash the stream.

## Tests

```bash
uv run pytest
```

Snapshot-parity tests live in `tests/snapshots/` and are generated from the upstream TS test fixtures.
