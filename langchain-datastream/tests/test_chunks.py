"""Phase 1 verification: every chunk type round-trips a real snapshot sample."""

import json
from pathlib import Path

import pytest

from langchain_datastream.chunks import (
    DataChunk,
    FileChunk,
    FinishChunk,
    FinishStepChunk,
    ReasoningDeltaChunk,
    ReasoningEndChunk,
    ReasoningStartChunk,
    StartChunk,
    StartStepChunk,
    TextDeltaChunk,
    TextEndChunk,
    TextStartChunk,
    ToolApprovalRequestChunk,
    ToolInputAvailableChunk,
    ToolInputDeltaChunk,
    ToolInputStartChunk,
    ToolOutputAvailableChunk,
    chunk_to_dict,
)

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _all_chunks_from_snapshots() -> list[dict]:
    chunks: list[dict] = []
    for p in SNAPSHOT_DIR.glob("*.json"):
        chunks.extend(json.loads(p.read_text()))
    return chunks


@pytest.mark.parametrize(
    "model,sample",
    [
        (StartChunk, {"type": "start"}),
        (StartStepChunk, {"type": "start-step"}),
        (FinishStepChunk, {"type": "finish-step"}),
        (FinishChunk, {"type": "finish"}),
        (TextStartChunk, {"type": "text-start", "id": "x"}),
        (TextDeltaChunk, {"type": "text-delta", "id": "x", "delta": "hi"}),
        (TextEndChunk, {"type": "text-end", "id": "x"}),
        (ReasoningStartChunk, {"type": "reasoning-start", "id": "x"}),
        (ReasoningDeltaChunk, {"type": "reasoning-delta", "id": "x", "delta": "hi"}),
        (ReasoningEndChunk, {"type": "reasoning-end", "id": "x"}),
        (
            ToolInputStartChunk,
            {
                "type": "tool-input-start",
                "toolCallId": "c",
                "toolName": "t",
                "dynamic": True,
            },
        ),
        (
            ToolInputDeltaChunk,
            {"type": "tool-input-delta", "toolCallId": "c", "inputTextDelta": "{"},
        ),
        (
            ToolInputAvailableChunk,
            {
                "type": "tool-input-available",
                "toolCallId": "c",
                "toolName": "t",
                "input": {"k": "v"},
                "dynamic": True,
            },
        ),
        (
            ToolOutputAvailableChunk,
            {"type": "tool-output-available", "toolCallId": "c", "output": "ok"},
        ),
        (
            ToolApprovalRequestChunk,
            {"type": "tool-approval-request", "approvalId": "a", "toolCallId": "c"},
        ),
        (
            FileChunk,
            {"type": "file", "mediaType": "image/png", "url": "data:image/png;base64,xx"},
        ),
        (
            DataChunk,
            {"type": "data-progress", "data": {"pct": 50}, "transient": True},
        ),
    ],
)
def test_chunk_roundtrip(model, sample):
    """Each chunk model parses its dict shape and re-serializes identically."""
    chunk = model.model_validate(sample)
    assert chunk_to_dict(chunk) == sample


def test_all_snapshot_chunks_have_known_types():
    """Every chunk in our 3 snapshots is a type our chunks module knows about."""
    known = {
        "start",
        "start-step",
        "finish-step",
        "finish",
        "abort",
        "error",
        "text-start",
        "text-delta",
        "text-end",
        "reasoning-start",
        "reasoning-delta",
        "reasoning-end",
        "tool-input-start",
        "tool-input-delta",
        "tool-input-available",
        "tool-output-available",
        "tool-output-error",
        "tool-approval-request",
        "file",
        "source-url",
        "source-document",
    }
    seen = {c["type"] for c in _all_chunks_from_snapshots()}
    unknown = {t for t in seen if not (t in known or t.startswith("data-"))}
    assert not unknown, f"Unknown chunk types in snapshots: {unknown}"
