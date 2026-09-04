"""--fast 로그가 기계·시각과 무관하게 같은지 본다.

공개 재생 페이지(site/)는 커밋된 비행 데이터가 지금 코드가 만들어내는 것과
같다는 전제 위에 서 있다. 그 전제가 깨지면 CI 가 매번 빨간불이 되거나, 더
나쁘게는 페이지가 실제로 돌지 않는 비행을 보여주게 된다.

한때 `started = time.monotonic()` 을 fast 모드에서도 썼다. started 가 크면
started + virtual 의 반올림 폭도 커져서 4.0초 같은 경계를 넘는 시점이 한 틱
밀렸고, 오래 켜둔 기계와 갓 부팅한 러너가 서로 다른 로그를 냈다.
"""

import dataclasses
import contextlib
import io
import os
import tempfile
import time
import unittest

from drone import main as main_mod
from drone.config import DEFAULT


def fly(monotonic_base: float) -> list[dict]:
    """time.monotonic() 이 주어진 값 근처를 돌려줄 때의 비행 로그."""
    frames: list[dict] = []
    real = time.monotonic
    with tempfile.TemporaryDirectory() as tmp:
        cfg = dataclasses.replace(
            DEFAULT,
            runtime=dataclasses.replace(
                DEFAULT.runtime, log_path=os.path.join(tmp, "log.jsonl")
            ),
        )
        main_mod.time.monotonic = lambda: monotonic_base
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                main_mod.run(
                    cfg, fast=True, max_seconds=60.0,
                    on_tick=lambda rec, *a: frames.append(dict(rec)),
                )
        finally:
            main_mod.time.monotonic = real
    return frames


class FastLogIsReproducible(unittest.TestCase):
    def test_uptime_does_not_change_the_flight(self):
        """갓 부팅한 러너와 오래 켜둔 컨테이너가 같은 로그를 내야 한다."""
        fresh = fly(42.0)              # 부팅 42초 뒤
        old = fly(987654.321)          # 11일 켜둔 기계

        self.assertEqual(len(fresh), len(old), "틱 수가 가동 시간에 따라 달라진다")
        self.assertEqual(fresh, old)

    def test_two_runs_in_a_row_match(self):
        self.assertEqual(fly(1000.0), fly(1000.0))

    def test_recorded_time_starts_at_zero(self):
        """t 는 실제 시각이 아니라 이륙 후 경과 시간이어야 한다."""
        self.assertEqual(fly(987654.321)[0]["t"], 0.0)


if __name__ == "__main__":
    unittest.main()
