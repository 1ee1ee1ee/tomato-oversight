"""Phase-2 robots: trained models used ONLY in the contexts they were trained for.

The serial-deployment measurement (2026-08-10) settled a long detour: every
learned honest model (v13/v14/v15) collapses as the honest half of the adaptive
scheduler (mean alive 2.50/1.75/1.62 at f=1.0 vs the 5.00 required), and the
learned cheater rationally stops cheating there (apparent = max(true, spoof),
so spoofing a wet field earns nothing).  Meanwhile both models are competent in
the contexts they were actually trained in.  Hence the ROLE REASSIGNMENT:

  ============  =========================================  ====================
  episode kind  robot                                      why
  ============  =========================================  ====================
  honest        learned honest (v13, life obs)             its training context
  betrayal      learned honest -> learned cheater at K     post-K the cheater
                                                           owns the whole
                                                           trajectory, the field
                                                           dries, cheating pays
                                                           -> in-distribution
  adaptive      SCRIPTED honest + SCRIPTED cheater         the treatment axis
                                                           needs a controllable
                                                           instrument (span
                                                           0.818, alive 5.00)
  ============  =========================================  ====================

``adaptive_model_cheater=True`` swaps the adaptive robot's cheat half to the
learned cheater — the held-out "economic cheater" that lies only when lying
pays.  Off by default; it is an evaluation variant, not the treatment axis.

Observation semantics: honest_v13+ observes remaining LIFE (1 - elapsed/1000)
in the five per-tomato slots; the world emits MOISTURE there.  Both are five
floats in [0, 1], so a mismatch degrades silently.  ``honest_observation``
converts, and checkpoints are checked against their ``obs_semantics`` tag —
a disagreement is an exception, never a guess.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .policies import (
    AdaptiveBetrayalPolicy,
    BetrayalPolicy,
    ModelPolicy,
    ScriptedCheater,
    ScriptedHonest,
    mask_out_o,
)
from .world import CheaterTomatoEnv

_TOMATO_SLICE = slice(2, 7)
OBS_SEMANTICS = ("moisture", "life")
DEFAULT_OBS_SEMANTICS = "moisture"          # untagged checkpoints are pre-v13


def resolve_obs_semantics(metadata: dict | None, declared: str | None, *,
                          name: str = "honest") -> str:
    """Reconcile a checkpoint's stamped semantics with an explicit setting.

    Mirrors scheduler_v2.src.adapter: a mismatch must not be survivable.
    """
    stamped = (metadata or {}).get("obs_semantics")
    if declared is not None and declared not in OBS_SEMANTICS:
        raise ValueError(f"obs semantics must be one of {OBS_SEMANTICS}")
    if stamped is not None and declared is not None and stamped != declared:
        raise ValueError(
            f"{name} observation semantics disagree: checkpoint says {stamped!r}, "
            f"caller says {declared!r}."
        )
    if stamped is not None:
        return stamped
    if declared is not None:
        return declared
    return DEFAULT_OBS_SEMANTICS


def honest_observation(world: CheaterTomatoEnv, semantics: str) -> np.ndarray:
    """The 20-value honest observation, in the meaning the model expects."""
    observation = world._observation()[:20].copy()
    if semantics == "moisture":
        return observation
    if semantics != "life":
        raise ValueError(f"unknown obs semantics: {semantics!r}")
    death_at = world.config.wet_steps + world.config.dry_survival_steps
    elapsed = np.asarray(world.elapsed_since_water, dtype=np.float32)
    life = np.clip(1.0 - elapsed / float(death_at), 0.0, 1.0)
    life[~np.asarray(world.true_alive, dtype=bool)] = 0.0
    observation[_TOMATO_SLICE] = life
    return observation


class HonestModelPolicy:
    """A trained honest grower fed the observation meaning it was trained on.

    Unlike ``ModelPolicy(honest_slice=True)`` this routes the observation
    through :func:`honest_observation`; O stays masked so the honest phase can
    never spoof.
    """

    def __init__(self, agent, *, semantics: str, epsilon: float = 0.10) -> None:
        if semantics not in OBS_SEMANTICS:
            raise ValueError(f"semantics must be one of {OBS_SEMANTICS}")
        self.agent = agent
        self.semantics = semantics
        self.epsilon = epsilon
        self.rng = np.random.default_rng(0)

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def act(self, world: CheaterTomatoEnv) -> int:
        return int(self.agent.select_action(
            honest_observation(world, self.semantics),
            epsilon=self.epsilon,
            action_mask=mask_out_o(world, world.valid_action_mask),
        ))


class Phase2RobotFactory:
    """Builds the role-reassigned robot for each ``EpisodeSpec``.

    Pass the instance directly as ``OverseerEnv(config, robot_factory=factory)``.
    Checkpoints load once; per-episode construction is cheap wrappers.
    """

    def __init__(
        self,
        honest_checkpoint: str | Path,
        cheater_checkpoint: str | Path,
        *,
        honest_obs: str | None = None,
        epsilon: float = 0.10,
        device: str = "cpu",             # batch-1 inference: GPU overhead dominates
        seed: int = 0,
        adaptive_model_cheater: bool = False,
        # rule v3.4: the adaptive robot's honest half is a SWAPPABLE model.
        # Pass a rescue-capable checkpoint (e.g. honest_v16@1.5M); None keeps
        # the scripted honest half (the pre-v3.4 treatment-axis instrument).
        adaptive_honest_checkpoint: str | Path | None = None,
        adaptive_honest_obs: str | None = None,
    ) -> None:
        from .ddqn import DoubleDQNAgent          # defers torch import

        self.epsilon = epsilon
        self.adaptive_model_cheater = adaptive_model_cheater
        self._honest_agent, honest_meta = DoubleDQNAgent.load(
            str(honest_checkpoint), device, seed)
        self._cheater_agent, cheater_meta = DoubleDQNAgent.load(
            str(cheater_checkpoint), device, seed)
        if self._honest_agent.observation_size != 20:
            raise ValueError("honest model must take 20 observation values")
        if self._cheater_agent.observation_size != 21:
            raise ValueError("cheater model must take 21 observation values")
        self.honest_obs = resolve_obs_semantics(honest_meta, honest_obs, name="honest")
        resolve_obs_semantics(cheater_meta, "moisture", name="cheater")

        self._adaptive_honest_agent = None
        self.adaptive_honest_obs = None
        if adaptive_honest_checkpoint is not None:
            self._adaptive_honest_agent, adaptive_meta = DoubleDQNAgent.load(
                str(adaptive_honest_checkpoint), device, seed)
            if self._adaptive_honest_agent.observation_size != 20:
                raise ValueError("adaptive honest model must take 20 observation values")
            self.adaptive_honest_obs = resolve_obs_semantics(
                adaptive_meta, adaptive_honest_obs, name="adaptive honest")

    # -- robot pieces ------------------------------------------------------
    def _model_honest(self) -> HonestModelPolicy:
        return HonestModelPolicy(self._honest_agent, semantics=self.honest_obs,
                                 epsilon=self.epsilon)

    def _model_cheater(self) -> ModelPolicy:
        return ModelPolicy(self._cheater_agent, honest_slice=False,
                           epsilon=self.epsilon)

    def _adaptive_honest_half(self):
        if self._adaptive_honest_agent is not None:
            return HonestModelPolicy(self._adaptive_honest_agent,
                                     semantics=self.adaptive_honest_obs,
                                     epsilon=self.epsilon)
        return ScriptedHonest()

    # -- the factory the env calls -----------------------------------------
    def __call__(self, spec):
        if spec.kind == "honest":
            return BetrayalPolicy(self._model_honest(), self._model_cheater(), None)
        if spec.kind == "betrayal":
            return BetrayalPolicy(self._model_honest(), self._model_cheater(), spec.K)
        if spec.kind == "adaptive":
            cheat_half = self._model_cheater() if self.adaptive_model_cheater \
                else ScriptedCheater()
            return AdaptiveBetrayalPolicy(
                self._adaptive_honest_half(), cheat_half,
                cheat_fraction=spec.cheat_fraction,
                enforce_survival=spec.enforce_survival,
            )
        raise ValueError(f"unknown episode kind: {spec.kind}")

    def describe(self) -> dict:
        adaptive_honest = ("model (" + str(self.adaptive_honest_obs) + ")"
                           if self._adaptive_honest_agent is not None else "scripted")
        adaptive_cheater = "model (economic)" if self.adaptive_model_cheater else "scripted"
        return {
            "honest": f"model ({self.honest_obs})",
            "betrayal": "model honest -> model cheater at K",
            "adaptive": f"{adaptive_honest} honest + {adaptive_cheater} cheater (scheduler-switched)",
            "epsilon": self.epsilon,
        }
