"""실행 루프.

    python -m drone.main                      # 시뮬레이션 (하드웨어 불필요)
    python -m drone.main --fast               # 대기 없이 즉시 완주
    python -m drone.main --link mavlink --perception oakd   # 실기체
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

from . import link as link_mod
from . import perception as perception_mod
from .config import DEFAULT, Config, Runtime
from .guard import Guard
from .policy import Policy
from .state import Phase


def build_config(args) -> Config:
    runtime = dataclasses.replace(
        DEFAULT.runtime,
        perception_backend=args.perception,
        link_backend=args.link,
        mavlink_url=args.mavlink_url,
        log_path=args.log,
    )
    return dataclasses.replace(DEFAULT, runtime=runtime)


def run(cfg: Config, *, fast: bool = False, max_seconds: float = 240.0) -> dict:
    source = perception_mod.build(cfg)
    fc = link_mod.build(cfg)
    policy = Policy(cfg.mission, max_yaw_rate=cfg.limits.max_yaw_rate)
    guard = Guard(cfg.limits)

    dt = 1.0 / cfg.runtime.loop_hz
    started = time.monotonic()
    virtual = 0.0
    vetoed_ticks = 0
    ticks = 0

    log = open(cfg.runtime.log_path, "w", encoding="utf-8")
    try:
        while True:
            now = started + virtual if fast else time.monotonic()

            if hasattr(fc, "pump"):
                fc.pump()
            telem = fc.telemetry()
            if fast:
                telem = dataclasses.replace(telem, stamp=now)

            percep = source.read(telem)
            if fast and percep is not None:
                percep = dataclasses.replace(percep, stamp=now)

            intent = policy.step(telem, percep, now)
            result = guard.check(intent, telem, percep, now)
            if result.force_phase is not None:
                policy.force(result.force_phase, now)

            fc.send(result.command, dt=dt)

            ticks += 1
            if result.vetoes:
                vetoed_ticks += 1

            record = {
                "t": round(virtual if fast else now - started, 2),
                "phase": policy.phase.value,
                "action": result.command.action.value,
                "vx": round(result.command.vx, 3),
                "vy": round(result.command.vy, 3),
                "vz": round(result.command.vz, 3),
                "yaw_rate": round(result.command.yaw_rate, 3),
                "n": round(telem.north_m, 2),
                "e": round(telem.east_m, 2),
                "alt": round(telem.alt_m, 2),
                "batt": round(telem.battery_pct, 1),
                "vetoes": list(result.vetoes),
                "reason": result.command.reason,
            }
            log.write(json.dumps(record, ensure_ascii=False) + "\n")

            if ticks % 5 == 0 or result.vetoes:
                flag = ("  ⚠ " + ",".join(result.vetoes)) if result.vetoes else ""
                print(
                    f"[{record['t']:6.1f}s] {policy.phase.value:<9}"
                    f" {result.command.action.value:<8}"
                    f" pos=({record['n']:+.2f},{record['e']:+.2f}) alt={record['alt']:.2f}"
                    f" batt={record['batt']:.0f}%  {result.command.reason}{flag}"
                )

            if policy.phase is Phase.DONE:
                break

            virtual += dt
            elapsed = virtual if fast else time.monotonic() - started
            if elapsed > max_seconds:
                print("최대 실행 시간 초과 — 중단")
                break

            if not fast:
                time.sleep(max(0.0, dt - (time.monotonic() - now)))
    finally:
        log.close()
        source.close()
        fc.close()

    summary = {
        "ticks": ticks,
        "vetoed_ticks": vetoed_ticks,
        "final_phase": policy.phase.value,
        "log": cfg.runtime.log_path,
    }
    print(
        f"\n완료: {ticks} tick, Guard 개입 {vetoed_ticks}회"
        f" ({vetoed_ticks / max(1, ticks) * 100:.1f}%), 최종 단계 {summary['final_phase']}"
    )
    print(f"비행 로그: {cfg.runtime.log_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="실내 온보드 자율 드론")
    p.add_argument("--perception", default="mock", choices=["mock", "oakd", "onnx"])
    p.add_argument("--link", default="mock", choices=["mock", "mavlink"])
    p.add_argument("--mavlink-url", default=Runtime().mavlink_url)
    p.add_argument("--log", default=Runtime().log_path)
    p.add_argument("--fast", action="store_true", help="실시간 대기 없이 실행")
    p.add_argument("--max-seconds", type=float, default=240.0)
    args = p.parse_args()

    run(build_config(args), fast=args.fast, max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
