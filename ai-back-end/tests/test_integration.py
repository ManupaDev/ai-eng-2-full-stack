"""Integration test: hit /api/chat with a real LLM call, verify SSE protocol.

Skipped when OPENAI_API_KEY is unset so unit-test runs stay hermetic.
Marked `slow` because it makes a real network call.
"""

import json
import os

import httpx
import pytest

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="needs OPENAI_API_KEY in .env.local",
    ),
    pytest.mark.slow,
]


async def _post_chat(payload):
    """POST against the live ASGI app (not over the network — uses ASGITransport)."""
    from main import server

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server),
        base_url="http://test",
        timeout=60.0,
    ) as client:
        async with client.stream("POST", "/api/chat", json=payload) as resp:
            assert resp.status_code == 200, await resp.aread()
            assert resp.headers["x-vercel-ai-ui-message-stream"] == "v1"
            assert resp.headers["content-type"].startswith("text/event-stream")
            chunks = []
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload_str = line[len("data: ") :]
                if payload_str == "[DONE]":
                    chunks.append({"_done": True})
                    continue
                chunks.append(json.loads(payload_str))
            return chunks


async def test_chat_streams_full_lifecycle():
    chunks = await _post_chat(
        {
            "messages": [
                {
                    "id": "u1",
                    "role": "user",
                    "parts": [{"type": "text", "text": "Reply with exactly one short sentence: hello"}],
                }
            ]
        }
    )

    types = [c.get("type") for c in chunks if "type" in c]

    # Must begin with start, end with finish, and have a [DONE] terminator.
    assert types[0] == "start", f"first chunk: {chunks[0]}"
    assert "finish" in types
    assert chunks[-1] == {"_done": True}, f"last chunk: {chunks[-1]}"

    # Should have at least one text-start / text-delta / text-end triplet.
    # (astream_events streams tokens, so multiple text-deltas are expected.)
    assert "text-start" in types
    assert "text-delta" in types
    assert "text-end" in types
    assert types.count("text-delta") >= 1

    # Concatenated text should be non-empty.
    text = "".join(c["delta"] for c in chunks if c.get("type") == "text-delta")
    assert text.strip(), f"empty model response: {chunks}"


async def test_chat_text_lifecycle_invariant():
    """Every text-delta must be flanked by text-start and text-end with the same id."""
    chunks = await _post_chat(
        {
            "messages": [
                {
                    "id": "u1",
                    "role": "user",
                    "parts": [{"type": "text", "text": "say hi"}],
                }
            ]
        }
    )

    open_text_ids: set[str] = set()
    for c in chunks:
        t = c.get("type")
        if t == "text-start":
            open_text_ids.add(c["id"])
        elif t == "text-delta":
            assert c["id"] in open_text_ids, f"orphan text-delta: {c}"
        elif t == "text-end":
            assert c["id"] in open_text_ids
            open_text_ids.discard(c["id"])
