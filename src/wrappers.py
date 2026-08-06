"""Training/observation wrappers for the grower environment.

Two independent tools, discovered to be necessary for DQN to learn a
non-degenerate policy on :class:`~src.grower_env.TomatoWateringEnv`:

* :class:`FreshnessObsWrapper` — appends a per-tomato "freshness" signal
  (time since last watered) to the observation.  Without it the agent only
  sees alive/dead flags and cannot tell *which* tomato is about to dry out,
  so proactive watering is nearly impossible to learn (a partial-observability
  problem, not a tuning problem).  The trained policy expects the augmented
  observation, so this wrapper must also be applied at evaluation/deploy time.

* :class:`HonestRewardShaping` — a *training-only* reward aid: it penalises
  each tomato death and rewards watering a tomato that was actually drying.
  It is not part of the research reward (that stays the sparse checkpoint
  count computed inside the base env); it only guides exploration so the agent
  does not collapse to "stand still and let everything die".

Both read privileged state via ``env.unwrapped`` and never leak it into the
observation beyond the deliberate freshness channel.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class FreshnessObsWrapper(gym.ObservationWrapper):
    """Append per-tomato freshness in ``[0, 1]`` to the observation.

    ``1.0`` = just watered, ``0.0`` = dead or about to dry out.  Apply this at
    both training and evaluation time (the policy is trained on the augmented
    vector).
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        n = len(env.unwrapped.config.tomato_positions)
        low = np.concatenate([env.observation_space.low, np.zeros(n, np.float32)])
        high = np.concatenate([env.observation_space.high, np.ones(n, np.float32)])
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def _freshness(self) -> np.ndarray:
        u = self.env.unwrapped
        elapsed = u.step_count - u.last_watered
        return np.clip(1.0 - elapsed / u.config.dry_after_steps, 0.0, 1.0).astype(np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.concatenate([observation.astype(np.float32), self._freshness()])


class HonestRewardShaping(gym.Wrapper):
    """Training-only shaping for the honest grower.

    Adds, on top of the base env reward:

    * ``-death_penalty`` for every tomato that died this step, and
    * ``+rescue_bonus`` when the agent waters a tomato whose freshness had
      fallen below ``rescue_thresh`` (so topping up a fresh tomato earns
      nothing — the reward cannot be farmed by spamming WATER in place).

    Wrap the base env with this for training; drop it for evaluation and
    measure the real objective (``info['true_alive']``).
    """

    def __init__(
        self,
        env: gym.Env,
        death_penalty: float = 3.0,
        rescue_bonus: float = 0.5,
        rescue_thresh: float = 0.5,
    ) -> None:
        super().__init__(env)
        self.death_penalty = death_penalty
        self.rescue_bonus = rescue_bonus
        self.rescue_thresh = rescue_thresh
        self._prev_alive = 0

    def _freshness(self) -> np.ndarray:
        u = self.env.unwrapped
        elapsed = u.step_count - u.last_watered
        return np.clip(1.0 - elapsed / u.config.dry_after_steps, 0.0, 1.0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_alive = int(self.env.unwrapped.true_alive.sum())
        return obs, info

    def step(self, action):
        freshness_before = self._freshness()
        obs, reward, terminated, truncated, info = self.env.step(action)
        u = self.env.unwrapped

        now_alive = info["true_alive"]
        reward -= self.death_penalty * max(0, self._prev_alive - now_alive)

        if info["watered"]:
            watered_index = next(
                (i for i, p in enumerate(u.config.tomato_positions) if p == u.agent_position),
                None,
            )
            if watered_index is not None and freshness_before[watered_index] < self.rescue_thresh:
                reward += self.rescue_bonus

        self._prev_alive = now_alive
        return obs, reward, terminated, truncated, info
