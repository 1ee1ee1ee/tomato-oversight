"""Evaluate a trained honest-grower model at all four O-wall positions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.ddqn import DoubleDQNAgent
from src.env import Action, HonestGrowerConfig, HonestTomatoEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes-per-o", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--show-first-grid", action="store_true")
    args = parser.parse_args()

    agent, metadata = DoubleDQNAgent.load(args.model, args.device, args.seed)
    config = HonestGrowerConfig()
    rows = []
    episode = 0
    for o_position in config.o_candidates:
        for repeat in range(args.episodes_per_o):
            env = HonestTomatoEnv(config)
            observation, _ = env.reset(
                seed=args.seed + episode,
                options={"o_position": o_position},
            )
            counts: Counter[str] = Counter()
            total_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.select_action(observation, epsilon=0.0)
                counts[Action(action).name] += 1
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
            if args.show_first_grid and episode == 0:
                print(env.render())
            rows.append(
                {
                    "episode": episode,
                    "o_position": o_position,
                    "reward": total_reward,
                    "final_alive": info["true_alive"],
                    "final_dry": info["true_dry"],
                    "actions": dict(counts),
                }
            )
            episode += 1
            env.close()

    result = {
        "model": str(args.model),
        "training_metadata": metadata,
        "full_survival_rate": float(np.mean([row["final_alive"] == 5 for row in rows])),
        "mean_final_alive": float(np.mean([row["final_alive"] for row in rows])),
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
        "episodes": rows,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
