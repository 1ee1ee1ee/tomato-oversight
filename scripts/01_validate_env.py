"""Run a short, deterministic honest/cheating environment smoke test."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow direct execution with ``python scripts/01_validate_env.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.grower_env import Action, GrowerConfig, TomatoWateringEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("honest", "cheating"), default="cheating")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true", help="open the Tkinter human-render window")
    parser.add_argument("--steps", type=int, default=18, help="number of demonstration actions")
    parser.add_argument("--delay-ms", type=int, default=250, help="pause between rendered actions")
    args = parser.parse_args()

    env = TomatoWateringEnv(
        GrowerConfig(mode=args.mode, max_steps=max(20, args.steps + 1)),
        render_mode="human" if args.render else None,
    )
    observation, info = env.reset(seed=args.seed)
    print(f"mode={args.mode} seed={args.seed} observation={observation.tolist()}")
    print(f"initial_info={info}")
    if args.render:
        env.render()
    else:
        print(env.render())

    if args.render:
        print("Close the environment window to end the demo.")
        env.wait_for_close()
    env.close()


if __name__ == "__main__":
    main()
