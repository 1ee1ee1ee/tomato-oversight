"""공개 재생 페이지에 실을 비행 데이터를 만든다.

    python3 site/build.py            # site/flight_data.json 을 새로 쓴다
    python3 site/build.py --check    # 새로 쓰지 않고, 커밋된 것과 다른지만 본다

site/orders.txt 의 명령을 한 줄씩 컴파일하고, 통과한 것만 시뮬레이터로
끝까지 비행시켜 프레임을 모은다. 판정(통과/거부)은 여기서 정하지 않는다 —
drone/mission_spec.py 의 validate() 가 내린 결과를 그대로 싣는다. 그래서
누가 orders.txt 에 이상한 명령을 추가해도 페이지가 거짓말을 하지 않는다.

하드웨어도 네트워크도 쓰지 않는다. mock 인지 + mock 링크 = 순수 계산.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone.compiler import compile_mission          # noqa: E402
from drone.config import DEFAULT                    # noqa: E402
from drone.main import run                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ORDERS = os.path.join(HERE, "orders.txt")
OUT = os.path.join(HERE, "flight_data.json")

# 재생용으로 5Hz 로 솎는다. 10Hz 로그를 그대로 실으면 파일이 두 배가 되는데
# 화면에서 보이는 것은 똑같다.
KEEP_EVERY = 2
MAX_SECONDS = 90.0

# 페이지가 그리는 필드만 남긴다. vx/vy/vz/yaw_rate 는 지도에 안 나온다.
FIELDS = ("t", "phase", "action", "n", "e", "alt", "batt", "vetoes", "reason")


def read_orders(path: str = ORDERS) -> list[str]:
    orders: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                orders.append(line)
    if not orders:
        raise SystemExit(f"{path} 에 명령이 하나도 없다")
    return orders


def fly(spec) -> list[dict]:
    """한 임무를 끝까지 시뮬레이션하고 프레임 목록을 돌려준다."""
    frames: list[dict] = []

    def on_tick(record, telem, percep, result):
        frames.append({k: record[k] for k in FIELDS})

    # run() 은 로그 파일을 쓰고 진행 상황을 stdout 에 찍는다. 빌드 중에는
    # 둘 다 필요 없다 — 임시 파일로 보내고 출력은 삼킨다.
    with tempfile.TemporaryDirectory() as tmp:
        cfg = dataclasses.replace(
            DEFAULT,
            runtime=dataclasses.replace(
                DEFAULT.runtime, log_path=os.path.join(tmp, "flight_log.jsonl")
            ),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            run(cfg, fast=True, max_seconds=MAX_SECONDS, spec=spec, on_tick=on_tick)

    return frames[::KEEP_EVERY]


def build() -> dict:
    limits = DEFAULT.limits
    scenarios = []

    for order in read_orders():
        res = compile_mission(order, limits, backend="rules")
        spec = res.spec
        scenarios.append({
            "order": order,
            "behavior": spec.behavior.value,
            "target": spec.target_label,
            "stop_distance_m": spec.stop_distance_m,
            "cruise_alt_m": spec.cruise_alt_m,
            "stop_after_n": spec.stop_after_n,
            "notes": list(res.notes),
            "problems": list(res.problems),
            "accepted": res.ok,
            # 거부된 명령은 비행하지 않는다. 빈 로그가 곧 그 사실의 증거다.
            "log": fly(spec) if res.ok else [],
        })

    return {
        "limits": dataclasses.asdict(limits),
        "scenarios": scenarios,
    }


def dumps(data: dict) -> str:
    """시나리오마다 한 줄. 통째로 한 줄이면 diff 에서 뭐가 바뀌었는지 안 보인다."""
    lines = [json.dumps(s, ensure_ascii=False, separators=(",", ":"))
             for s in data["scenarios"]]
    limits = json.dumps(data["limits"], ensure_ascii=False, separators=(",", ":"))
    return ('{"limits":' + limits + ',\n "scenarios":[\n  '
            + ",\n  ".join(lines) + "\n ]}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="공개 재생 페이지 데이터 빌드")
    p.add_argument(
        "--check", action="store_true",
        help="파일을 쓰지 않고 커밋된 결과와 다른지만 검사한다 (CI 용)",
    )
    args = p.parse_args()

    text = dumps(build())

    if args.check:
        try:
            with open(OUT, encoding="utf-8") as f:
                committed = f.read()
        except FileNotFoundError:
            committed = ""
        if committed != text:
            print(
                "site/flight_data.json 이 orders.txt / 시뮬레이터와 어긋난다.\n"
                "  python3 site/build.py 를 돌리고 결과를 같이 커밋하세요.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print("site/flight_data.json 최신 상태")
        return

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)

    data = json.loads(text)
    for s in data["scenarios"]:
        mark = "통과" if s["accepted"] else "거부"
        print(f"  {mark}  {s['order']}  ({len(s['log'])} 프레임)")
    print(f"\n{OUT} — {os.path.getsize(OUT) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
