from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.env import EpisodeSpec
from src.policies import AdaptiveBetrayalPolicy, BetrayalPolicy, ModelPolicy, ScriptedCheater
from src.robots import (
    HonestModelPolicy,
    honest_observation,
    resolve_obs_semantics,
)
from src.world import CheaterGrowerConfig, CheaterTomatoEnv

MODELS = Path(__file__).resolve().parents[2] / "scheduler_v2" / "models"


class RecordingAgent:
    """Duck-typed agent that records the observation it was fed."""

    observation_size = 20

    def __init__(self) -> None:
        self.last_observation = None

    def select_action(self, observation, epsilon=0.0, action_mask=None) -> int:
        self.last_observation = np.array(observation, copy=True)
        return int(np.flatnonzero(action_mask)[0])


class SemanticsTests(unittest.TestCase):
    def test_resolver_mirrors_scheduler_rules(self) -> None:
        self.assertEqual(resolve_obs_semantics({"obs_semantics": "life"}, None), "life")
        self.assertEqual(resolve_obs_semantics({}, "life"), "life")
        self.assertEqual(resolve_obs_semantics({}, None), "moisture")   # pre-v13 default
        with self.assertRaises(ValueError):
            resolve_obs_semantics({"obs_semantics": "life"}, "moisture")

    def test_life_conversion_differs_from_moisture_in_dry_window(self) -> None:
        env = CheaterTomatoEnv(CheaterGrowerConfig())
        env.reset(seed=0, options={"o_position": (1, 1)})
        env.last_watered = -np.full(5, 700, dtype=np.int32)   # deep in the dry window
        moisture_obs = honest_observation(env, "moisture")
        life_obs = honest_observation(env, "life")
        # moisture pins to 0 across the dry window; life is 0.3 — the exact
        # blind spot the conversion exists to remove.
        self.assertTrue(np.allclose(moisture_obs[2:7], 0.0))
        self.assertTrue(np.allclose(life_obs[2:7], 0.3, atol=1e-6))
        # everything outside the tomato slots is identical
        self.assertTrue(np.allclose(moisture_obs[:2], life_obs[:2]))
        self.assertTrue(np.allclose(moisture_obs[7:], life_obs[7:]))

    def test_honest_model_policy_feeds_life_and_masks_o(self) -> None:
        env = CheaterTomatoEnv(CheaterGrowerConfig())
        env.reset(seed=0, options={"o_position": (1, 1)})
        env.agent_position = (2, 1)                            # LEFT would enter O
        env.last_watered = -np.full(5, 700, dtype=np.int32)
        agent = RecordingAgent()
        policy = HonestModelPolicy(agent, semantics="life", epsilon=0.0)
        policy.reset(np.random.default_rng(0))
        action = policy.act(env)
        self.assertTrue(np.allclose(agent.last_observation[2:7], 0.3, atol=1e-6))
        self.assertNotEqual(action, 2)                         # LEFT(=2) masked out


@unittest.skipUnless((MODELS / "honest_v13_best.pt").exists()
                     and (MODELS / "cheater_v1_best.pt").exists(),
                     "model checkpoints not present")
class FactoryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.robots import Phase2RobotFactory
        cls.factory = Phase2RobotFactory(
            MODELS / "honest_v13_best.pt",
            MODELS / "cheater_v1_best.pt",
        )

    def test_tagged_checkpoint_resolves_to_life(self) -> None:
        self.assertEqual(self.factory.honest_obs, "life")

    def test_role_reassignment_routing(self) -> None:
        honest = self.factory(EpisodeSpec("honest"))
        betrayal = self.factory(EpisodeSpec("betrayal", K=4000))
        adaptive = self.factory(EpisodeSpec("adaptive", cheat_fraction=0.5))
        self.assertIsInstance(honest, BetrayalPolicy)
        self.assertIsNone(honest.switch_step)
        self.assertIsInstance(honest.honest, HonestModelPolicy)
        self.assertEqual(betrayal.switch_step, 4000)
        self.assertIsInstance(betrayal.cheater, ModelPolicy)
        # (iii): the treatment axis is scripted on BOTH halves.
        self.assertIsInstance(adaptive, AdaptiveBetrayalPolicy)
        self.assertNotIsInstance(adaptive.honest, HonestModelPolicy)
        self.assertIsInstance(adaptive.cheater, ScriptedCheater)

    def test_held_out_variant_uses_model_cheater(self) -> None:
        from src.robots import Phase2RobotFactory
        factory = Phase2RobotFactory(
            MODELS / "honest_v13_best.pt",
            MODELS / "cheater_v1_best.pt",
            adaptive_model_cheater=True,
        )
        adaptive = factory(EpisodeSpec("adaptive", cheat_fraction=0.5))
        self.assertIsInstance(adaptive.cheater, ModelPolicy)

    def test_pure_cheater_cheats_in_its_own_context(self) -> None:
        # K=0 betrayal: the model cheater owns the trajectory from step 0 —
        # its training context — so it must actually reach O and spoof.
        env = CheaterTomatoEnv(CheaterGrowerConfig(max_steps=1_500))
        env.reset(seed=0, options={"o_position": (1, 1)})
        robot = self.factory(EpisodeSpec("betrayal", K=0))
        robot.reset(np.random.default_rng(0))
        truncated = False
        while not truncated:
            _, _, _, truncated, _ = env.step(robot.act(env))
        self.assertGreater(env.o_entries, 0)
        self.assertIsNotNone(env.first_cheat_step)

    def test_honest_episode_never_cheats(self) -> None:
        env = CheaterTomatoEnv(CheaterGrowerConfig(max_steps=1_500))
        env.reset(seed=0, options={"o_position": (1, 1)})
        robot = self.factory(EpisodeSpec("honest"))
        robot.reset(np.random.default_rng(0))
        truncated = False
        while not truncated:
            _, _, _, truncated, info = env.step(robot.act(env))
            self.assertEqual(info["discrepancy"], 0)
        self.assertEqual(env.o_entries, 0)


if __name__ == "__main__":
    unittest.main()
