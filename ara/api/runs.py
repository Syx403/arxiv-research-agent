"""A turn runs as a server task, not inside the request that started it (D38). Its events are kept
in order, so any number of followers (the page that sent it, the same page after a reload, a page
that switched away and back) can read them from the start; a dropped connection leaves the turn
running, and only `stop` ends it early."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Run:
    conversation: str
    turn_id: str
    message: str
    kind: str  # "messages" or "resume"
    events: list[tuple[str, Any]] = field(default_factory=list)
    finished: bool = False
    task: asyncio.Task[None] | None = None
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)

    async def emit(self, event: str, data: Any) -> None:
        async with self.changed:
            self.events.append((event, data))
            self.changed.notify_all()

    async def finish(self) -> None:
        async with self.changed:
            self.finished = True
            self.changed.notify_all()

    async def follow(self) -> AsyncIterator[tuple[str, Any]]:
        """Every event so far, then each new one, until the turn ends."""
        at = 0
        while True:
            async with self.changed:
                while at == len(self.events) and not self.finished:
                    await self.changed.wait()
                ready, done = self.events[at:], self.finished
            for event in ready:
                yield event
            at += len(ready)
            if done and at == len(self.events):
                return

    def summary(self) -> dict[str, str]:
        return {"turn_id": self.turn_id, "message": self.message, "kind": self.kind}
