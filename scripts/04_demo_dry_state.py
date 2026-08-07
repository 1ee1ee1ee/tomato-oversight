"""Open the human renderer exactly when all unwatered tomatoes become dry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.grower_env import Action, GrowerConfig, TomatoWateringEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--moisture-steps", type=int, default=500)
    parser.add_argument("--dry-survival-steps", type=int, default=500)
    args = parser.parse_args()

    env = TomatoWateringEnv(
        GrowerConfig(
            mode="honest",
            moisture_depletion_steps=args.moisture_steps,
            dry_survival_steps=args.dry_survival_steps,
        ),
        render_mode="human",
    )
    env.reset(seed=args.seed)
    info = None
    for _ in range(args.moisture_steps):
        _, _, _, _, info = env.step(Action.WAIT)

    print(
        f"step={info['step']} true_alive={info['true_alive']} "
        f"true_dry={info['true_dry']} dry_steps={info['dry_steps']}"
    )
    env.render()
    env.wait_for_close()
    env.close()


if __name__ == "__main__":
    main()
