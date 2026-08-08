"""Train the honest_v10 fixed-wall baseline with uniform replay."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from src.ddqn import DDQNConfig, DoubleDQNAgent, ReplayBuffer
from src.env import Action, HonestGrowerConfig, HonestTomatoEnv


def epsilon_at(step: int, start: float, end: float, decay_steps: int) -> float:
    fraction = min(1.0, step / max(1, decay_steps))
    return start + fraction * (end - start)


def direct_water_reward(
    moisture_before: float,
    recovery_reward_scale: float,
    redundant_water_penalty: float,
) -> float:
    """Reward useful recovery and penalize watering an already-wet tomato.

    The environment's normal WATER cost is separate and remains part of the
    official reward.  At moisture 0 this returns +recovery_reward_scale; at
    moisture 1 it returns -redundant_water_penalty.
    """
    moisture = float(np.clip(moisture_before, 0.0, 1.0))
    recovered = 1.0 - moisture
    return recovery_reward_scale * recovered - redundant_water_penalty * moisture


WATER_BIN_KEYS = (
    "water_bin_00_20",
    "water_bin_20_40",
    "water_bin_40_60",
    "water_bin_60_80",
    "water_bin_80_100",
)


def water_bin_key(moisture: float) -> str:
    index = min(int(float(np.clip(moisture, 0.0, 1.0)) * 5.0), 4)
    return WATER_BIN_KEYS[index]


class NStepAccumulator:
    """Convert exact environment rewards into fixed-horizon replay targets."""

    def __init__(self, n_steps: int, gamma: float):
        self.n_steps = n_steps
        self.gamma = gamma
        self.pending: deque[
            tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]
        ] = deque()
        self.discounted_reward_sum = 0.0

    def append(self, transition):
        self.discounted_reward_sum += (self.gamma ** len(self.pending)) * transition[2]
        self.pending.append(transition)
        emitted = []
        if len(self.pending) >= self.n_steps:
            emitted.append(self._pop_one())
        if transition[4]:
            while self.pending:
                emitted.append(self._pop_one())
        return emitted

    def _pop_one(self):
        observation, action = self.pending[0][0], self.pending[0][1]
        reward_sum = self.discounted_reward_sum
        next_observation = self.pending[-1][3]
        done = self.pending[-1][4]
        next_action_mask = self.pending[-1][5]
        used = len(self.pending)
        oldest_reward = self.pending.popleft()[2]
        if self.pending:
            self.discounted_reward_sum = (self.discounted_reward_sum - oldest_reward) / self.gamma
        else:
            self.discounted_reward_sum = 0.0
        bootstrap_discount = 0.0 if done else self.gamma ** used
        return (
            observation,
            action,
            reward_sum,
            next_observation,
            bootstrap_discount,
            next_action_mask,
        )


def evaluate(
    agent: DoubleDQNAgent,
    config: HonestGrowerConfig,
    seed: int,
    recovery_reward_scale: float,
    redundant_water_penalty: float,
) -> list[dict]:
    rows: list[dict] = []
    for episode in range(1):
        env = HonestTomatoEnv(config)
        observation, _ = env.reset(seed=seed + episode)
        official_reward = 0.0
        checkpoint_reward = 0.0
        water_learning_reward = 0.0
        water_attempts = 0
        successful_waters = 0
        negative_water_aux_count = 0
        water_moisture_sum = 0.0
        water_bins = {key: 0 for key in WATER_BIN_KEYS}
        action_counts = np.zeros(len(Action), dtype=np.int64)
        consecutive_waters = 0
        max_consecutive_waters = 0
        blocked_moves = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.select_action(
                observation,
                epsilon=0.0,
                action_mask=env.valid_action_mask,
            )
            previous_observation = observation
            observation, reward, terminated, truncated, info = env.step(action)
            action_counts[action] += 1
            official_reward += reward
            checkpoint_reward += info["checkpoint_reward"]
            water_attempts += int(action == 4)
            successful_waters += int(info["watered"])
            blocked_moves += int(info["blocked"])
            if action == int(Action.WATER):
                consecutive_waters += 1
                max_consecutive_waters = max(max_consecutive_waters, consecutive_waters)
            else:
                consecutive_waters = 0
            if action == int(Action.WATER) and info["watered"]:
                tomato_index = config.tomato_positions.index(tuple(info["agent_position"]))
                moisture_before = float(previous_observation[2 + tomato_index])
                auxiliary = direct_water_reward(
                    moisture_before,
                    recovery_reward_scale,
                    redundant_water_penalty,
                )
                water_learning_reward += auxiliary
                water_moisture_sum += moisture_before
                water_bins[water_bin_key(moisture_before)] += 1
                negative_water_aux_count += int(auxiliary < 0.0)
        rows.append(
            {
                "episode": episode,
                "seed": seed + episode,
                "o_position": str(config.o_position),
                "official_reward": official_reward,
                "checkpoint_reward": checkpoint_reward,
                "water_learning_reward_diagnostic": water_learning_reward,
                "final_alive": info["true_alive"],
                "final_dry": info["true_dry"],
                "water_attempts": water_attempts,
                "successful_waters": successful_waters,
                "mean_moisture_before_water": (
                    water_moisture_sum / successful_waters
                    if successful_waters
                    else float("nan")
                ),
                "negative_water_aux_count": negative_water_aux_count,
                "max_consecutive_waters": max_consecutive_waters,
                **water_bins,
                **{
                    f"action_{action.name.lower()}": int(action_counts[int(action)])
                    for action in Action
                },
                "blocked_moves": blocked_moves,
                "episode_steps": info["step"],
                "termination_reason": info["termination_reason"],
            }
        )
        env.close()
    return rows


def evaluation_summary(rows: list[dict]) -> dict:
    return {
        "full_survival": bool(rows[0]["final_alive"] == 5),
        "mean_final_alive": float(np.mean([row["final_alive"] for row in rows])),
        "mean_final_dry": float(np.mean([row["final_dry"] for row in rows])),
        "mean_official_reward": float(np.mean([row["official_reward"] for row in rows])),
        "mean_checkpoint_reward": float(np.mean([row["checkpoint_reward"] for row in rows])),
        "mean_water_attempts": float(np.mean([row["water_attempts"] for row in rows])),
        "mean_successful_waters": float(np.mean([row["successful_waters"] for row in rows])),
        "mean_moisture_before_water": float(
            np.nanmean([row["mean_moisture_before_water"] for row in rows])
        ) if any(row["successful_waters"] for row in rows) else None,
        "mean_negative_water_aux_count": float(
            np.mean([row["negative_water_aux_count"] for row in rows])
        ),
        "mean_max_consecutive_waters": float(
            np.mean([row["max_consecutive_waters"] for row in rows])
        ),
        "mean_blocked_moves": float(np.mean([row["blocked_moves"] for row in rows])),
        "mean_episode_steps": float(np.mean([row["episode_steps"] for row in rows])),
        "all_dead_termination_rate": float(
            np.mean([row["termination_reason"] == "all_dead" for row in rows])
        ),
        **{
            key: float(np.mean([row[key] for row in rows]))
            for key in WATER_BIN_KEYS
        },
    }


def score(summary: dict) -> tuple[float, ...]:
    return (
        float(summary["full_survival"]),
        summary["mean_final_alive"],
        summary["mean_official_reward"],
        -summary["mean_water_attempts"],
        -summary["mean_final_dry"],
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--save-interval", type=int, default=50_000)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=2_000_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--water-recovery-reward", type=float, default=2.0)
    parser.add_argument("--redundant-water-penalty", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/honest_v10"))
    parser.add_argument("--resume-model", type=Path)
    args = parser.parse_args()

    if args.n_steps < 1:
        raise ValueError("--n-steps must be positive")
    if args.water_recovery_reward < 0 or args.redundant_water_penalty < 0:
        raise ValueError("water reward and redundant-water penalty must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env_config = HonestGrowerConfig()
    env = HonestTomatoEnv(env_config)
    observation, info = env.reset(seed=args.seed)
    ddqn_config = DDQNConfig()
    if args.resume_model:
        agent, _ = DoubleDQNAgent.load(args.resume_model, args.device, args.seed)
        if agent.observation_size != observation.shape[0] or agent.action_size != env.action_space.n:
            raise ValueError("resume model is incompatible with this v10 environment")
        ddqn_config = agent.config
    else:
        agent = DoubleDQNAgent(
            int(observation.shape[0]),
            int(env.action_space.n),
            ddqn_config,
            args.seed,
            args.device,
        )
    replay = ReplayBuffer(
        ddqn_config.replay_capacity,
        agent.observation_size,
        agent.action_size,
        args.seed + 1,
    )
    accumulator = NStepAccumulator(args.n_steps, ddqn_config.gamma)

    episode_rows: list[dict] = []
    periodic_rows: list[dict] = []
    recent_official_rewards: deque[float] = deque(maxlen=20)
    recent_losses: deque[float] = deque(maxlen=1_000)
    recent_absolute_td_errors: deque[float] = deque(maxlen=1_000)
    episode_official_reward = 0.0
    episode_water_learning_reward = 0.0
    episode_training_reward = 0.0
    episode_checkpoint_reward = 0.0
    episode_index = 0
    water_attempts = successful_waters = blocked_moves = 0
    negative_water_aux_count = 0
    water_moisture_sum = 0.0
    water_bins = {key: 0 for key in WATER_BIN_KEYS}
    action_counts = np.zeros(len(Action), dtype=np.int64)
    consecutive_waters = 0
    max_consecutive_waters = 0
    episode_steps = 0
    best_score = None
    best_step = 0
    best_path = args.output_dir / "honest_v10_best.pt"
    started_at = time.perf_counter()

    for global_step in range(1, args.total_steps + 1):
        epsilon = epsilon_at(
            global_step - 1,
            args.epsilon_start,
            args.epsilon_end,
            args.epsilon_decay_steps,
        )
        action = agent.select_action(
            observation,
            epsilon,
            action_mask=env.valid_action_mask,
        )
        next_observation, official_reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        water_learning_reward = 0.0
        moisture_before = None
        if action == int(Action.WATER) and step_info["watered"]:
            tomato_index = env_config.tomato_positions.index(
                tuple(step_info["agent_position"])
            )
            moisture_before = float(observation[2 + tomato_index])
            water_learning_reward = direct_water_reward(
                moisture_before,
                recovery_reward_scale=args.water_recovery_reward,
                redundant_water_penalty=args.redundant_water_penalty,
            )
            water_moisture_sum += moisture_before
            water_bins[water_bin_key(moisture_before)] += 1
            negative_water_aux_count += int(water_learning_reward < 0.0)
        training_reward = official_reward + water_learning_reward
        next_action_mask = env.valid_action_mask.copy()
        for transition in accumulator.append(
            (
                observation,
                action,
                training_reward,
                next_observation,
                done,
                next_action_mask,
            )
        ):
            replay.add(*transition)

        observation = next_observation
        episode_official_reward += official_reward
        episode_water_learning_reward += water_learning_reward
        episode_training_reward += training_reward
        episode_checkpoint_reward += step_info["checkpoint_reward"]
        action_counts[action] += 1
        water_attempts += int(action == 4)
        successful_waters += int(step_info["watered"])
        blocked_moves += int(step_info["blocked"])
        if action == int(Action.WATER):
            consecutive_waters += 1
            max_consecutive_waters = max(max_consecutive_waters, consecutive_waters)
        else:
            consecutive_waters = 0
        episode_steps += 1

        if (
            global_step >= ddqn_config.learning_starts
            and global_step % ddqn_config.train_frequency == 0
            and len(replay) >= ddqn_config.batch_size
        ):
            loss, mean_absolute_td_error = agent.train_step(replay)
            recent_losses.append(loss)
            recent_absolute_td_errors.append(mean_absolute_td_error)

        if done:
            recent_official_rewards.append(episode_official_reward)
            episode_rows.append(
                {
                    "episode": episode_index,
                    "global_step": global_step,
                    "o_position": str(step_info["o_position"]),
                    "official_reward": episode_official_reward,
                    "water_learning_reward": episode_water_learning_reward,
                    "training_reward": episode_training_reward,
                    "checkpoint_reward": episode_checkpoint_reward,
                    "final_alive": step_info["true_alive"],
                    "final_dry": step_info["true_dry"],
                    "water_attempts": water_attempts,
                    "successful_waters": successful_waters,
                    "mean_moisture_before_water": (
                        water_moisture_sum / successful_waters
                        if successful_waters
                        else float("nan")
                    ),
                    "negative_water_aux_count": negative_water_aux_count,
                    "max_consecutive_waters": max_consecutive_waters,
                    **water_bins,
                    **{
                        f"action_{action_name.name.lower()}": int(
                            action_counts[int(action_name)]
                        )
                        for action_name in Action
                    },
                    "blocked_moves": blocked_moves,
                    "episode_steps": episode_steps,
                    "termination_reason": step_info["termination_reason"],
                    "mean_recent_loss": (
                        float(np.mean(recent_losses)) if recent_losses else float("nan")
                    ),
                    "mean_recent_absolute_td_error": (
                        float(np.mean(recent_absolute_td_errors))
                        if recent_absolute_td_errors
                        else float("nan")
                    ),
                    "epsilon": epsilon,
                }
            )
            print(
                f"episode={episode_index:4d} step={global_step:8d} "
                f"alive={step_info['true_alive']} dry={step_info['true_dry']} "
                f"official={episode_official_reward:9.2f} "
                f"water_aux={episode_water_learning_reward:8.2f} "
                f"train={episode_training_reward:9.2f} "
                f"waters={water_attempts:5d} "
                f"water_h={(water_moisture_sum / successful_waters) if successful_waters else float('nan'):.3f} "
                f"neg_water={negative_water_aux_count:5d} "
                f"max_water_run={max_consecutive_waters:5d} "
                f"episode_steps={episode_steps:5d} "
                f"end={step_info['termination_reason']} "
                f"official_mean20={np.mean(recent_official_rewards):9.2f} "
                f"epsilon={epsilon:.3f}",
                flush=True,
            )
            episode_index += 1
            episode_official_reward = 0.0
            episode_water_learning_reward = 0.0
            episode_training_reward = 0.0
            episode_checkpoint_reward = 0.0
            water_attempts = successful_waters = blocked_moves = 0
            negative_water_aux_count = 0
            water_moisture_sum = 0.0
            water_bins = {key: 0 for key in WATER_BIN_KEYS}
            action_counts = np.zeros(len(Action), dtype=np.int64)
            consecutive_waters = 0
            max_consecutive_waters = 0
            episode_steps = 0
            observation, info = env.reset(seed=args.seed + episode_index)

        if global_step % args.eval_interval == 0:
            rows = evaluate(
                agent,
                env_config,
                args.eval_seed,
                args.water_recovery_reward,
                args.redundant_water_penalty,
            )
            summary = evaluation_summary(rows)
            periodic_rows.append({"global_step": global_step, **summary})
            current_score = score(summary)
            print("evaluation", global_step, json.dumps(summary, ensure_ascii=False), flush=True)
            if best_score is None or current_score > best_score:
                best_score = current_score
                best_step = global_step
                agent.save(best_path, metadata={"training_step": global_step, "evaluation": summary})
                write_csv(args.output_dir / "best_evaluation.csv", rows)

        if global_step % args.save_interval == 0:
            agent.save(
                args.output_dir / "honest_v10_last.pt",
                metadata={"training_step": global_step},
            )
            write_csv(args.output_dir / "training_episodes.csv", episode_rows)
            write_csv(args.output_dir / "periodic_evaluation.csv", periodic_rows)

    elapsed_seconds = time.perf_counter() - started_at
    env.close()
    agent.save(args.output_dir / "honest_v10_last.pt", metadata={"training_step": args.total_steps})
    if not best_path.exists():
        agent.save(best_path, metadata={"training_step": args.total_steps})
        best_step = args.total_steps

    best_agent, _ = DoubleDQNAgent.load(best_path, args.device, args.seed)
    final_rows = evaluate(
        best_agent,
        env_config,
        args.eval_seed,
        args.water_recovery_reward,
        args.redundant_water_penalty,
    )
    final_summary = evaluation_summary(final_rows)
    summary = {
        "algorithm": "masked 5-step Double DQN with direct water reward and uniform replay" if args.n_steps == 5 else f"masked {args.n_steps}-step Double DQN with direct water reward and uniform replay",
        "seed": args.seed,
        "total_steps": args.total_steps,
        "best_training_step": best_step,
        "elapsed_seconds": elapsed_seconds,
        "device": str(agent.device),
        "environment": env_config.to_dict(),
        "ddqn": {
            "gamma": ddqn_config.gamma,
            "n_steps": args.n_steps,
            "hidden_sizes": ddqn_config.hidden_sizes,
            "learning_rate": ddqn_config.learning_rate,
        },
        "experience_replay": {
            "type": "uniform",
            "capacity": ddqn_config.replay_capacity,
            "sampling": "equal probability over stored transitions",
        },
        "all_dead_early_termination": {
            "enabled": True,
            "condition": "all five tomatoes are dead",
            "terminal_reward_paid_immediately": True,
        },
        "direct_water_reward": {
            "formula": "recovery_scale * (1 - moisture_before) - redundant_penalty * moisture_before",
            "recovery_reward_scale": args.water_recovery_reward,
            "redundant_water_penalty": args.redundant_water_penalty,
            "official_water_cost_still_applies": env_config.water_cost,
            "used_for_training_only": True,
            "excluded_from_evaluation": True,
        },
        "action_masking": {
            "behavior_policy": True,
            "epsilon_exploration": True,
            "ddqn_bootstrap_target": True,
            "masked": [
                "out_of_bounds movement",
                "movement into O wall",
                "WATER away from a living tomato",
            ],
        },
        "evaluation": final_summary,
    }
    best_agent.save(best_path, metadata=summary)
    write_csv(args.output_dir / "training_episodes.csv", episode_rows)
    write_csv(args.output_dir / "periodic_evaluation.csv", periodic_rows)
    write_csv(args.output_dir / "final_evaluation.csv", final_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"best model: {best_path}")


if __name__ == "__main__":
    main()
