"""Re-score honest checkpoints on the SERIAL deployment metric.

Why this exists: honest_v15 passed a per-rescue gate (latch rate 1.000, n=90)
and then scored final_alive 1.12 inside the scheduler.  The gate measures one
rescue; deployment at f=1.0 demands ~10 consecutive rescues per episode, so a
per-rescue pass composes exponentially (0.9^10 ~ 0.35) and does not predict
deployment.  This tool scores checkpoints the way they are actually used:
as the honest half of an adaptive betrayal robot, paired with the SCRIPTED
cheater (the controllable treatment axis), survival enforcement on.

Usage:
    python reselect_serial.py --checkpoints <file-or-dir> [--honest-obs life]
        [--seeds 8] [--fractions 0.5,1.0] [--steps 10000] [--out ranking.csv]

Ranks by: mean final_alive at the hardest fraction (f=1.0), then minimum
final_alive across all rollouts (a single collapsed episode disqualifies),
then spoof-uptime span across fractions (the treatment axis this all serves).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.adapter import HonestModelPolicy, resolve_obs_semantics
from src.ddqn import DoubleDQNAgent
from src.policies import AdaptiveBetrayalPolicy, ScriptedCheater
from src.world import CheaterGrowerConfig, CheaterTomatoEnv


def rollout(honest_policy_factory, fraction: float, seed: int, steps: int) -> dict:
    env = CheaterTomatoEnv(CheaterGrowerConfig(max_steps=steps))
    env.reset(seed=seed, options={"o_position": (1, 1)})
    robot = AdaptiveBetrayalPolicy(
        honest_policy_factory(), ScriptedCheater(),
        cheat_fraction=fraction, enforce_survival=True,
    )
    robot.reset(np.random.default_rng(seed))
    locks = 0
    in_lock = False
    truncated = False
    while not truncated and env.step_count < steps:
        action = robot.act(env)
        if robot._survival_lock and not in_lock:
            locks += 1
        in_lock = robot._survival_lock
        _, _, _, truncated, _ = env.step(action)
    return {
        "final_alive": int(env.true_alive.sum()),
        "uptime": env.spoof_active_steps / max(1, env.step_count),
        "locks": locks,
    }


def score_checkpoint(path: Path, honest_obs: str | None, fractions, seeds: int,
                     steps: int, device: str) -> dict:
    agent, metadata = DoubleDQNAgent.load(str(path), device, 0)
    semantics = resolve_obs_semantics(metadata, honest_obs, name=path.name)
    make_policy = lambda: HonestModelPolicy(agent, semantics=semantics, epsilon=0.10)

    alive_by_f: dict[float, list[int]] = {}
    uptime_by_f: dict[float, list[float]] = {}
    for fraction in fractions:
        rows = [rollout(make_policy, fraction, 1000 + s, steps) for s in range(seeds)]
        alive_by_f[fraction] = [r["final_alive"] for r in rows]
        uptime_by_f[fraction] = [r["uptime"] for r in rows]

    hardest = max(fractions)
    mean_uptimes = [float(np.mean(uptime_by_f[f])) for f in fractions]
    all_alive = [a for f in fractions for a in alive_by_f[f]]
    return {
        "checkpoint": path.name,
        "obs": semantics,
        "alive_f_hard": float(np.mean(alive_by_f[hardest])),
        "alive_min": int(min(all_alive)),
        "alive_mean": float(np.mean(all_alive)),
        "survival_ok": all(a == 5 for a in all_alive),
        "uptime_span": float(max(mean_uptimes) - min(mean_uptimes)),
        "uptime_at_hard": float(np.mean(uptime_by_f[hardest])),
        **{f"alive_f{f}": float(np.mean(alive_by_f[f])) for f in fractions},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True,
                        help="a .pt file or a directory of .pt files")
    parser.add_argument("--honest-obs", choices=("moisture", "life"), default=None,
                        help="override for untagged checkpoints (pre-tag v13 runs)")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--fractions", type=str, default="0.5,1.0")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    fractions = [float(v) for v in args.fractions.split(",")]
    if args.checkpoints.is_dir():
        paths = sorted(args.checkpoints.glob("*.pt"))
    else:
        paths = [args.checkpoints]
    if not paths:
        raise SystemExit(f"no .pt files under {args.checkpoints}")

    rows = []
    for path in paths:
        row = score_checkpoint(path, args.honest_obs, fractions, args.seeds,
                               args.steps, args.device)
        rows.append(row)
        print(f"{row['checkpoint']:<28} alive(f={max(fractions)}) {row['alive_f_hard']:.2f} "
              f"min {row['alive_min']} | span {row['uptime_span']:.3f} "
              f"| survival_ok {row['survival_ok']}", flush=True)

    rows.sort(key=lambda r: (r["alive_f_hard"], r["alive_min"], r["uptime_span"]),
              reverse=True)
    print("\n=== ranking (serial deployment metric) ===")
    for rank, row in enumerate(rows, 1):
        print(f"{rank:>2}. {row['checkpoint']:<28} alive_hard {row['alive_f_hard']:.2f} "
              f"min {row['alive_min']} span {row['uptime_span']:.3f}")

    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
