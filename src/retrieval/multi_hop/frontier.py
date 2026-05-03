from __future__ import annotations

import heapq
import itertools

from src.core.types import FrontierItem


class Frontier:
    def __init__(self) -> None:
        self._heap: list[tuple[float, int, FrontierItem]] = []
        self._counter = itertools.count()
        self._visited: set[str] = set()
        self._enqueued: set[str] = set()

    def add(self, item: FrontierItem) -> bool:
        if item.paper_id in self._visited or item.paper_id in self._enqueued:
            return False
        heapq.heappush(self._heap, (-item.score, next(self._counter), item))
        self._enqueued.add(item.paper_id)
        return True

    def pop(self) -> FrontierItem | None:
        if not self._heap:
            return None
        _, _, item = heapq.heappop(self._heap)
        return item

    def mark_visited(self, paper_id: str) -> None:
        self._visited.add(paper_id)

    @property
    def visited(self) -> set[str]:
        return set(self._visited)

    @property
    def enqueued(self) -> set[str]:
        return set(self._enqueued)

    def __len__(self) -> int:
        return len(self._heap)
