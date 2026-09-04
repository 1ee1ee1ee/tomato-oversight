"""관제 대시보드 테스트.

Hub 는 비행 루프(자기 스레드)와 HTTP 핸들러(각자 다른 스레드)가 공유하는
유일한 상태다. 여기가 틀리면 화면이 조용히 옛날 값을 보여준다.
"""

import dataclasses
import json
import queue
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from drone.config import DEFAULT
from drone.dashboard.hub import SUBSCRIBER_BACKLOG, Hub, MissionRecord
from drone.dashboard.server import Console


def tick(t=0.0, phase="search", n=0.0, e=0.0, vetoes=()) -> dict:
    return {"t": t, "phase": phase, "action": "move", "n": n, "e": e,
            "alt": 1.3, "batt": 90.0, "vetoes": list(vetoes), "reason": ""}


class HubBasics(unittest.TestCase):
    def setUp(self):
        self.hub = Hub()

    def test_snapshot_starts_empty(self):
        s = self.hub.snapshot()
        self.assertEqual(s["latest"], {})
        self.assertIsNone(s["mission"])
        self.assertFalse(s["flying"])

    def test_publish_updates_latest(self):
        self.hub.publish(tick(t=1.0, phase="approach"))
        self.assertEqual(self.hub.snapshot()["latest"]["phase"], "approach")

    def test_track_collects_positions(self):
        self.hub.publish(tick(n=1.0, e=2.0))
        self.hub.publish(tick(n=1.5, e=2.5))
        self.assertEqual(self.hub.track(), [[1.0, 2.0], [1.5, 2.5]])

    def test_history_is_bounded(self):
        """화면용 이력이 무한히 자라면 장시간 비행에서 메모리를 먹는다."""
        for i in range(2600):
            self.hub.publish(tick(t=i))
        self.assertLessEqual(self.hub.snapshot()["ticks"], 2000)

    def test_reset_clears_screen_state(self):
        self.hub.publish(tick(t=1.0))
        self.hub.reset()
        self.assertEqual(self.hub.snapshot()["latest"], {})
        self.assertEqual(self.hub.track(), [])


class HubSubscribers(unittest.TestCase):
    def setUp(self):
        self.hub = Hub()

    def test_subscriber_receives_published(self):
        q = self.hub.subscribe()
        self.hub.publish(tick(t=3.0))
        self.assertEqual(q.get_nowait()["t"], 3.0)

    def test_unsubscribed_stops_receiving(self):
        q = self.hub.subscribe()
        self.hub.unsubscribe(q)
        self.hub.publish(tick())
        with self.assertRaises(queue.Empty):
            q.get_nowait()

    def test_slow_subscriber_drops_oldest_not_newest(self):
        """관제 화면은 최신 상태가 중요하다. 밀리면 옛 것을 버려야 한다."""
        q = self.hub.subscribe()
        for i in range(SUBSCRIBER_BACKLOG + 20):
            self.hub.publish(tick(t=float(i)))
        received = []
        while True:
            try:
                received.append(q.get_nowait()["t"])
            except queue.Empty:
                break
        self.assertLessEqual(len(received), SUBSCRIBER_BACKLOG)
        # 마지막 틱이 살아 있어야 한다
        self.assertEqual(received[-1], float(SUBSCRIBER_BACKLOG + 19))

    def test_publish_does_not_hold_lock_while_feeding(self):
        """구독자에게 밀어넣는 동안 락을 쥐고 있으면 느린 클라이언트가
        비행 루프를 멈춰 세운다. 다른 스레드가 동시에 읽을 수 있어야 한다."""
        self.hub.subscribe()
        done = threading.Event()

        def reader():
            for _ in range(200):
                self.hub.snapshot()
            done.set()

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        for i in range(200):
            self.hub.publish(tick(t=float(i)))
        t.join(timeout=5)
        self.assertTrue(done.is_set())


class HubStopFlag(unittest.TestCase):
    def test_stop_is_sticky_until_flight_ends(self):
        hub = Hub()
        self.assertFalse(hub.should_stop())
        hub.request_stop()
        self.assertTrue(hub.should_stop())
        hub.set_flying(False)          # 비행이 끝나면 다음 임무를 위해 풀린다
        self.assertFalse(hub.should_stop())


class MissionRecordShape(unittest.TestCase):
    def test_serialises_for_the_browser(self):
        r = MissionRecord(order="페트병 2개", behavior="count", target="bottle",
                          notes=("메모",), problems=(), accepted=True)
        d = r.as_dict()
        self.assertEqual(d["behavior"], "count")
        self.assertEqual(d["notes"], ["메모"])
        self.assertTrue(d["accepted"])
        json.dumps(d)                  # 직렬화 가능해야 한다


class ServerRoutes(unittest.TestCase):
    """실제 HTTP 로 띄워서 확인한다. 라우팅 오타는 유닛테스트로 안 잡힌다."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cfg = dataclasses.replace(
            DEFAULT,
            runtime=dataclasses.replace(
                DEFAULT.runtime, log_path=str(Path(cls.tmp.name) / "log.jsonl")
            ),
        )
        cls.hub = Hub()
        cls.server = Console(("127.0.0.1", 0), cfg, cls.hub, fast=True)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def _get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as r:
            return r.status, r.read()

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_index_serves_html(self):
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"<title>", body)

    def test_limits_endpoint(self):
        status, body = self._get("/api/limits")
        self.assertEqual(status, 200)
        self.assertIn("max_radius_m", json.loads(body))

    def test_unknown_route_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/api/nope")
        self.assertEqual(ctx.exception.code, 404)

    def test_empty_order_is_rejected(self):
        status, body = self._post("/api/order", {"order": "   "})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_impossible_order_is_refused_without_flying(self):
        """프로펠러가 돌기 전에 거른다 — 화면에서도 그대로 보여야 한다."""
        status, body = self._post("/api/order", {"order": "사람 앞 30cm 까지 가"})
        self.assertEqual(status, 200)
        self.assertFalse(body["accepted"])
        self.assertTrue(any("전방 회피 여유" in p for p in body["mission"]["problems"]))
        self.assertFalse(self.hub.snapshot()["flying"])

    def test_valid_order_starts_a_flight(self):
        status, body = self._post("/api/order", {"order": "사람 찾아서 앞에 서줘"})
        self.assertEqual(status, 200)
        self.assertTrue(body["accepted"])
        self.assertEqual(body["mission"]["behavior"], "approach_inspect")
        self.server._flight.join(timeout=30)
        self.assertGreater(self.hub.snapshot()["ticks"], 0)


if __name__ == "__main__":
    unittest.main()
