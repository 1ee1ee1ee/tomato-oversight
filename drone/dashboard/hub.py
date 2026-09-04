"""비행 루프 ↔ 대시보드 사이의 유일한 접점.

비행 루프는 자기 스레드에서 돌고 HTTP 핸들러는 각자 다른 스레드에서 돈다.
공유 상태를 여기 한 곳에 몰아넣고 락으로 감싼다 — 루프 코드에 스레드
관련 코드가 새어 들어가지 않게 하기 위해서다.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

#: SSE 구독자 한 명이 밀릴 때 버퍼링할 최대 틱 수.
#: 넘치면 오래된 것부터 버린다 — 관제 화면은 최신 상태가 중요하지
#: 모든 틱을 빠짐없이 받는 게 중요한 게 아니다.
SUBSCRIBER_BACKLOG = 64


@dataclass
class MissionRecord:
    """대시보드에 보여줄 임무 한 건의 요약."""

    order: str = ""
    behavior: str = ""
    target: str = ""
    notes: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()
    accepted: bool = False
    backend: str = "rules"
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "order": self.order,
            "behavior": self.behavior,
            "target": self.target,
            "notes": list(self.notes),
            "problems": list(self.problems),
            "accepted": self.accepted,
            "backend": self.backend,
            "at": self.at,
        }


class Hub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict = {}
        self._frame: bytes | None = None
        self._mission: MissionRecord | None = None
        self._flying = False
        self._stop = threading.Event()
        self._subscribers: list[queue.Queue] = []
        self._history: list[dict] = []

    # --- 비행 루프가 호출하는 쪽 -------------------------------------

    def publish(self, record: dict) -> None:
        """매 틱 상태를 밀어넣는다."""
        with self._lock:
            self._latest = record
            self._history.append(record)
            if len(self._history) > 2000:      # 로그는 파일에 남는다. 여긴 화면용.
                del self._history[:1000]
            subscribers = list(self._subscribers)

        for q in subscribers:
            try:
                q.put_nowait(record)
            except queue.Full:
                # 밀린 구독자는 오래된 것을 버리고 최신으로 따라잡는다.
                try:
                    q.get_nowait()
                    q.put_nowait(record)
                except (queue.Empty, queue.Full):
                    pass

    def set_frame(self, jpeg: bytes) -> None:
        with self._lock:
            self._frame = jpeg

    def set_flying(self, flying: bool) -> None:
        with self._lock:
            self._flying = flying
        if not flying:
            self._stop.clear()

    # --- 대시보드가 호출하는 쪽 ---------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "latest": dict(self._latest),
                "mission": self._mission.as_dict() if self._mission else None,
                "flying": self._flying,
                "ticks": len(self._history),
            }

    def latest_frame(self) -> bytes | None:
        with self._lock:
            return self._frame

    def track(self, limit: int = 600) -> list[list[float]]:
        """탑다운 맵에 그릴 비행 궤적 (north, east)."""
        with self._lock:
            rows = self._history[-limit:]
        return [[r.get("n", 0.0), r.get("e", 0.0)] for r in rows]

    def set_mission(self, record: MissionRecord) -> None:
        with self._lock:
            self._mission = record

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=SUBSCRIBER_BACKLOG)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    # --- 중단 ---------------------------------------------------------

    def request_stop(self) -> None:
        self._stop.set()

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def reset(self) -> None:
        """다음 임무를 위해 화면 상태를 비운다. 로그 파일은 건드리지 않는다."""
        with self._lock:
            self._history.clear()
            self._latest = {}
        self._stop.clear()
