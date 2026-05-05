"""Stream lifecycle callbacks.

Mirrors `/tmp/ai-sdk-repo/packages/langchain/src/stream-callbacks.ts`.
Each hook is optional and can be sync or async.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar

T = TypeVar("T")

OnStart = Callable[[], None | Awaitable[None]]
OnToken = Callable[[str], None | Awaitable[None]]
OnText = Callable[[str], None | Awaitable[None]]
OnFinal = Callable[[str], None | Awaitable[None]]
OnFinish = Callable[[Any], None | Awaitable[None]]
OnError = Callable[[BaseException], None | Awaitable[None]]
OnAbort = Callable[[], None | Awaitable[None]]


@dataclass
class StreamCallbacks(Generic[T]):
    """Lifecycle hooks for `to_ui_message_stream`.

    Attributes mirror the TS field names:
      - on_start  ↔ onStart
      - on_token  ↔ onToken (per text-delta)
      - on_text   ↔ onText (alias for on_token)
      - on_final  ↔ onFinal (full aggregated text at end)
      - on_finish ↔ onFinish (last LangGraph values payload, if any)
      - on_error  ↔ onError
      - on_abort  ↔ onAbort
    """

    on_start: OnStart | None = None
    on_token: OnToken | None = None
    on_text: OnText | None = None
    on_final: OnFinal | None = None
    on_finish: OnFinish | None = None
    on_error: OnError | None = None
    on_abort: OnAbort | None = None


async def maybe_await(result: Any) -> Any:
    """Await `result` if it's awaitable, otherwise return as-is."""
    if inspect.isawaitable(result):
        return await result
    return result
