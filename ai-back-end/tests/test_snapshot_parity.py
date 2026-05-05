"""Phase 6 — byte-for-byte parity with the TS adapter on real LangGraph fixtures.

Fixtures and snapshots are sourced verbatim from
`/tmp/ai-sdk-repo/packages/langchain/src/__fixtures__/langgraph.ts` and
`/tmp/ai-sdk-repo/packages/langchain/src/__snapshots__/*.json`.

If these tests pass we have full parity for the LangGraph stream-mode path
(the same one the TS test suite locks down).
"""

import json
from pathlib import Path

import pytest

from ai_sdk import to_ui_message_stream
from ai_sdk import langgraph_stream as lg

FIXTURES = Path(__file__).parent / "fixtures"
SNAPSHOTS = Path(__file__).parent / "snapshots"

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def deterministic_now(monkeypatch):
    """Mirror `vi.spyOn(Date, 'now').mockReturnValue(1234567890)` from adapter.test.ts."""
    monkeypatch.setattr(lg, "_now_ms", lambda: 1234567890)


async def _stream_to_chunks(fixture):
    async def _aiter(items):
        for item in items:
            yield item

    return [c async for c in to_ui_message_stream(_aiter(fixture))]


def _coerce_to_jsonable(chunks):
    """Normalize Python objects to JSON-equivalent form for comparison."""
    return json.loads(json.dumps(chunks))


@pytest.mark.parametrize(
    "fixture_name,snapshot_name",
    [
        ("LANGGRAPH_RESPONSE_1", "langgraph-hitl-request-1.json"),
        ("LANGGRAPH_RESPONSE_2", "langgraph-hitl-request-2.json"),
        ("REACT_AGENT_TOOL_CALLING", "react-agent-tool-calling.json"),
    ],
)
async def test_langgraph_snapshot_parity(fixture_name, snapshot_name):
    fixture = json.loads((FIXTURES / f"{fixture_name}.json").read_text())
    expected = json.loads((SNAPSHOTS / snapshot_name).read_text())

    chunks = await _stream_to_chunks(fixture)
    actual = _coerce_to_jsonable(chunks)

    if actual != expected:
        # Better diff for human inspection: show first divergence
        for i, (a, e) in enumerate(zip(actual, expected)):
            if a != e:
                pytest.fail(
                    f"First divergence at index {i}:\n"
                    f"  expected: {json.dumps(e)}\n"
                    f"  actual:   {json.dumps(a)}\n"
                    f"(actual length={len(actual)}, expected length={len(expected)})"
                )
        pytest.fail(
            f"Length mismatch: actual={len(actual)}, expected={len(expected)}\n"
            f"  trailing actual:   {json.dumps(actual[-3:])}\n"
            f"  trailing expected: {json.dumps(expected[-3:])}"
        )
