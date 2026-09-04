"""관제 대시보드.

비행 루프와 웹 화면 사이를 ``Hub`` 하나로 잇는다. 루프는 대시보드의
존재를 모르고, 대시보드는 루프의 내부를 모른다 — 둘 다 Hub 만 안다.

의존성 없음. ``http.server`` + SSE + MJPEG 로 표준 라이브러리만 쓴다.
"""

from .hub import Hub
from .server import serve

__all__ = ["Hub", "serve"]
