"""Reward shaping and n-step bookkeeping.  No torch — this is pure arithmetic.

Split out of ``train.py`` so the whole suite runs on a machine with only numpy
and gymnasium.  ``train.py`` imports torch at module scope, which used to drag
torch into every test that touched a potential or an n-step target, and this
project's training happens on Colab while its checking happens locally.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def epsilon_at(step: int, start: float, end: float, decay_steps: int) -> float:
    fraction = min(1.0, step / max(1, decay_steps))
    return start + fraction * (end - start)


def state_potential(
    observation: np.ndarray,
    tomato_count: int,
    life_weight: float,
    alive_weight: float,
) -> float:
    """Potential computed only from state values available to the honest policy.

    honest_v13: slots 2..2+n are remaining life, not moisture.  That is the
    whole point of the version — the old moisture slots were pinned at 0 for
    the entire dry window, so the potential had zero gradient exactly where
    "go water that one now" needed to be signalled.
    """
    life_start = 2
    alive_start = life_start + tomato_count
    life = observation[life_start:alive_start]
    alive = observation[alive_start:alive_start + tomato_count]
    return float(life_weight * life.sum() + alive_weight * alive.sum())


def potential_shaping_reward(
    observation: np.ndarray,
    next_observation: np.ndarray,
    done: bool,
    gamma: float,
    tomato_count: int,
    life_weight: float,
    alive_weight: float,
) -> float:
    current_potential = state_potential(
        observation, tomato_count, life_weight, alive_weight
    )
    next_potential = 0.0 if done else state_potential(
        next_observation, tomato_count, life_weight, alive_weight
    )
    return gamma * next_potential - current_potential


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
