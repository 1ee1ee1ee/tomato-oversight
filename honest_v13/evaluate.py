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
    # Match train.py: "1,1" evaluates the single fixed O, "cycle" all four.
    parser.add_argument("--o-position", type=str, default="1,1")
    # Deployment policy is epsilon-greedy (see train.py --eval-epsilon).
    parser.add_argument("--epsilon", type=float, default=0.10)
    args = parser.parse_args()

    agent, metadata = DoubleDQNAgent.load(args.model, args.device, args.seed)
    # Randomized starts are a training-robustness device; evaluation always
    # runs the standard step-0 start so numbers stay comparable with v11.
    config = HonestGrowerConfig(random_reset=False)
    if args.o_position == "cycle":
        eval_positions = list(config.o_candidates)
    else:
        fixed_o = tuple(int(value) for value in args.o_position.split(","))
        if fixed_o not in config.o_candidates:
            raise ValueError(f"--o-position must be 'cycle' or one of {config.o_candidates}")
        eval_positions = [fixed_o]
    rows = []
    episode = 0
    for o_position in eval_positions:
        for repeat in range(args.episodes_per_o):
            env = HonestTomatoEnv(config)
            observation, _ = env.reset(
                seed=args.seed + episode,
                options={"o_position": o_position},
            )
            counts: Counter[str] = Counter()
            official_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.select_action(
                    observation,
                    epsilon=args.epsilon,
                    action_mask=env.valid_action_mask,
                )
                counts[Action(action).name] += 1
                observation, reward, terminated, truncated, info = env.step(action)
                official_reward += reward
            if args.show_first_grid and episode == 0:
                print(env.render())
            rows.append(
                {
                    "episode": episode,
                    "o_position": o_position,
                    "official_reward": official_reward,
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
        "mean_official_reward": float(np.mean([row["official_reward"] for row in rows])),
        "episodes": rows,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
