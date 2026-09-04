"""관제 대시보드 HTTP 서버.

표준 라이브러리만 쓴다. 실시간 갱신은 WebSocket 대신 **SSE**(Server-Sent
Events)로 한다 — 그냥 HTTP 스트림이라 ``http.server`` 로 충분하고,
관제 화면은 서버→클라이언트 한 방향이면 되기 때문이다.

시연에서 중요한 점: 심사위원 앞에서 터미널에 명령어를 치면 그게 아무리
자연어여도 화면상으로는 코드 제어로 보인다. 이 화면은 한국어를 입력(또는
음성으로 말)받고, 컴파일 결과와 거부 사유까지 그대로 보여준다.
"""

from __future__ import annotations

import dataclasses
import hmac
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .. import main as main_mod
from ..compiler import compile_mission
from .hub import Hub, MissionRecord

INDEX = Path(__file__).with_name("index.html")

#: SSE 연결이 살아 있는지 확인하는 주석 프레임 주기(초).
#: 프록시가 조용한 연결을 끊는 것을 막는다.
HEARTBEAT_S = 15.0


class _Handler(BaseHTTPRequestHandler):
    server_version = "DroneConsole/1.0"

    # 기본 로거는 요청마다 stderr 에 찍어 비행 로그를 덮어버린다.
    def log_message(self, fmt, *args):
        return

    # --- 공통 응답 헬퍼 ------------------------------------------------

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, code: int = 200) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    # --- 라우팅 --------------------------------------------------------

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route in ("/", "/index.html"):
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif route == "/api/state":
            self._json(self.server.hub.snapshot())
        elif route == "/api/track":
            self._json({"track": self.server.hub.track()})
        elif route == "/api/limits":
            self._json(dataclasses.asdict(self.server.cfg.limits))
        elif route == "/api/stream":
            self._stream()
        elif route == "/api/camera":
            self._camera()
        else:
            self._json({"error": "not found"}, 404)

    def _authorised(self) -> bool:
        """쓰기 요청만 검사한다. 읽기는 열어둬야 화면이 그냥 뜬다."""
        token = self.server.token
        if not token:
            return True
        sent = self.headers.get("X-Console-Token") or ""
        # 길이가 달라도 상수 시간으로 비교한다.
        return hmac.compare_digest(sent, token)

    def do_POST(self):
        route = self.path.split("?", 1)[0]
        if not self._authorised():
            self._json({"error": "토큰이 필요합니다"}, 401)
            return
        if route == "/api/order":
            self._order()
        elif route == "/api/stop":
            self.server.hub.request_stop()
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    # --- 스트리밍 ------------------------------------------------------

    def _stream(self):
        """SSE. 매 틱 상태를 밀어준다."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        hub = self.server.hub
        q = hub.subscribe()
        last_beat = time.monotonic()
        try:
            # 붙자마자 현재 상태를 한 번 보내 화면이 비어 보이지 않게 한다.
            self._sse(hub.snapshot().get("latest") or {})
            while True:
                try:
                    record = q.get(timeout=1.0)
                    self._sse(record)
                except queue.Empty:
                    if time.monotonic() - last_beat > HEARTBEAT_S:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        last_beat = time.monotonic()
        except (BrokenPipeError, ConnectionResetError):
            pass          # 브라우저가 탭을 닫았다. 정상이다.
        finally:
            hub.unsubscribe(q)

    def _sse(self, payload: dict) -> None:
        self.wfile.write(b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n")
        self.wfile.flush()

    def _camera(self):
        """MJPEG. 프레임이 없으면 204 로 끝내 화면이 기다리지 않게 한다."""
        hub = self.server.hub
        if hub.latest_frame() is None:
            self._json({"error": "no camera"}, 204)
            return

        boundary = b"--frameboundary"
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frameboundary")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                frame = hub.latest_frame()
                if frame is not None:
                    self.wfile.write(boundary + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame + b"\r\n")
                    self.wfile.flush()
                time.sleep(1.0 / 15)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # --- 명령 접수 -----------------------------------------------------

    def _order(self):
        """자연어 명령 → 컴파일 → 검증 → (통과하면) 비행 시작.

        검증에 실패하면 비행을 시작하지 않고 사유만 돌려준다.
        프로펠러가 돌기 전에 거른다는 원칙이 화면에서도 그대로 보인다.
        """
        hub, cfg = self.server.hub, self.server.cfg
        payload = self._read_json()
        text = (payload.get("order") or "").strip()
        backend = payload.get("compiler") or cfg.runtime.compiler_backend

        if not text:
            self._json({"error": "명령이 비어 있습니다"}, 400)
            return
        if hub.snapshot()["flying"]:
            self._json({"error": "이미 비행 중입니다"}, 409)
            return

        try:
            result = compile_mission(text, cfg.limits, backend=backend)
        except Exception as exc:                     # 컴파일러 백엔드 장애
            self._json({"error": f"컴파일 실패: {exc}"}, 502)
            return

        record = MissionRecord(
            order=text,
            behavior=result.spec.behavior.value,
            target=result.spec.target_label,
            notes=result.notes,
            problems=result.problems,
            accepted=result.ok,
            backend=result.backend,
        )
        hub.set_mission(record)

        if not result.ok:
            self._json({"accepted": False, "mission": record.as_dict()})
            return

        hub.reset()
        self.server.start_flight(result.spec)
        self._json({"accepted": True, "mission": record.as_dict()})


class Console(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, cfg, hub: Hub, fast: bool = False,
                 token: str = "") -> None:
        super().__init__(address, _Handler)
        self.cfg = cfg
        self.hub = hub
        self.fast = fast
        #: 비어 있으면 인증 없음. 루프백 밖으로 열 때는 반드시 설정한다.
        self.token = token
        self._flight: threading.Thread | None = None

    def start_flight(self, spec) -> None:
        if self._flight is not None and self._flight.is_alive():
            return
        self.hub.set_flying(True)
        self._flight = threading.Thread(
            target=self._fly, args=(spec,), daemon=True, name="flight"
        )
        self._flight.start()

    def _fly(self, spec) -> None:
        try:
            main_mod.run(
                self.cfg,
                fast=self.fast,
                spec=spec,
                on_tick=lambda record, *_: self.hub.publish(record),
                should_stop=self.hub.should_stop,
            )
        except Exception as exc:                     # 비행 스레드가 조용히 죽지 않게
            self.hub.publish({"phase": "error", "reason": f"비행 루프 오류: {exc}"})
        finally:
            self.hub.set_flying(False)


LOOPBACK = ("127.0.0.1", "localhost", "::1")


def serve(
    cfg,
    host: str = "127.0.0.1",
    port: int = 8080,
    fast: bool = False,
    token: str = "",
) -> None:
    """대시보드를 띄운다.

    기본값은 **루프백 전용**이다. 이 서버의 ``POST /api/order`` 는 드론을
    이륙시킨다 — 네트워크에 열어두면 같은 망에 있는 누구나 기체를 띄울 수
    있다. 시연장 공용 WiFi 에서는 실제 위험이다.

    다른 기기(노트북 등)에서 봐야 하면 ``--host 0.0.0.0 --token <임의문자열>``
    로 명시적으로 열고, 토큰을 아는 사람만 명령을 보낼 수 있게 한다.
    인터넷에 직접 노출해서는 안 된다.
    """
    hub = Hub()

    if host not in LOOPBACK and not token:
        raise SystemExit(
            f"거부: {host} 로 열면서 토큰이 없습니다.\n"
            "  이 서버의 POST /api/order 는 드론을 이륙시킵니다. 같은 망에 있는\n"
            "  누구나 기체를 띄울 수 있게 됩니다.\n"
            "  --token <임의문자열> 을 붙이거나, --host 127.0.0.1 로 두세요."
        )

    console = Console((host, port), cfg, hub, fast=fast, token=token)
    shown = "localhost" if host in ("0.0.0.0", "") else host
    print(f"관제 대시보드: http://{shown}:{port}")
    if host not in LOOPBACK:
        print(f"⚠ {host} 로 열려 있습니다 — 같은 망의 다른 기기에서 접근됩니다.")
        print("  토큰이 설정돼 있으므로 명령 전송에는 토큰이 필요합니다.")
    print("브라우저에서 열고 한국어로 명령을 입력하세요. Ctrl+C 로 종료합니다.")
    try:
        console.serve_forever()
    except KeyboardInterrupt:
        print("\n종료합니다")
    finally:
        console.server_close()
