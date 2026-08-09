"""Baseline overseers (control group) and an episode runner.

An overseer policy is any object with ``reset()`` and ``act(obs, info) -> int``.
The learned DQN overseer (Phase 2) will implement the same interface, so the
comparison harness treats baselines and the trained agent identically.
"""

from __future__ import annotations

import numpy as np

from .env import OverseerAction, OverseerEnv


class PeriodicAuditor:
    """Control group: audit every ``period`` decisions, no matter what.

    ``period`` is in overseer *decisions*.  Use :func:`period_from_world_steps`
    to convert an "audit every N world steps" schedule.
    """

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def act(self, obs: np.ndarray, info: dict) -> int:
        self.count += 1
        audit = self.count % self.period == 0
        return int(OverseerAction.AUDIT if audit else OverseerAction.CONTINUE)


class NeverAudit:
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray, info: dict) -> int:
        return int(OverseerAction.CONTINUE)


class AlwaysAudit:
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray, info: dict) -> int:
        return int(OverseerAction.AUDIT)


def period_from_world_steps(n_steps: int, decision_interval: int) -> int:
    """'Audit every n_steps world steps' -> a decision period."""
    return max(1, round(n_steps / decision_interval))


def run_episode(env: OverseerEnv, policy, seed: int | None = None) -> dict:
    """Run one overseer episode; return the final info plus return/steps."""
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
    info = dict(info)
    info["return"] = total_reward
    info["decisions"] = steps
    return info
