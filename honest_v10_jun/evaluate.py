"""Evaluate a trained honest_v10 grower greedily at fixed O=(1, 1)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.ddqn import DoubleDQNAgent
from src.env import Action, HonestGrowerConfig, HonestTomatoEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--show-first-grid", action="store_true")
    args = parser.parse_args()

    agent, metadata = DoubleDQNAgent.load(args.model, args.device, args.seed)
    config = HonestGrowerConfig()
    env = HonestTomatoEnv(config)
    observation, _ = env.reset(seed=args.seed)
    counts: Counter[str] = Counter()
    official_reward = 0.0
    water_moistures: list[float] = []
    consecutive_waters = 0
    max_consecutive_waters = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.select_action(
            observation,
            epsilon=0.0,
            action_mask=env.valid_action_mask,
        )
        counts[Action(action).name] += 1
        previous_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)
        official_reward += reward
        if action == int(Action.WATER):
            consecutive_waters += 1
            max_consecutive_waters = max(max_consecutive_waters, consecutive_waters)
            if info["watered"]:
                index = config.tomato_positions.index(tuple(info["agent_position"]))
                water_moistures.append(float(previous_observation[2 + index]))
        else:
            consecutive_waters = 0
    if args.show_first_grid:
        print(env.render())
    row = {
        "o_position": config.o_position,
        "official_reward": official_reward,
        "final_alive": info["true_alive"],
        "final_dry": info["true_dry"],
        "episode_steps": info["step"],
        "termination_reason": info["termination_reason"],
        "water_attempts": counts[Action.WATER.name],
        "mean_moisture_before_water": (
            sum(water_moistures) / len(water_moistures)
            if water_moistures
            else None
        ),
        "max_consecutive_waters": max_consecutive_waters,
        "actions": dict(counts),
    }
    env.close()

    result = {
        "model": str(args.model),
        "training_metadata": metadata,
        "full_survival": row["final_alive"] == 5,
        "final_alive": row["final_alive"],
        "official_reward": row["official_reward"],
        "episode": row,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
