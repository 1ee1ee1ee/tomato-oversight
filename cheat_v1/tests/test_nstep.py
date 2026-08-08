from __future__ import annotations

import unittest

import numpy as np

from train import NStepAccumulator


class NStepTests(unittest.TestCase):
    def test_accumulates_discounted_rewards(self) -> None:
        accumulator = NStepAccumulator(n_steps=3, gamma=0.5)
        observation = np.zeros(2, dtype=np.float32)
        mask = np.ones(5, dtype=bool)
        emitted = []
        for reward in (1.0, 2.0, 4.0):
            emitted.extend(
                accumulator.append((observation, 0, reward, observation, False, mask))
            )
        self.assertEqual(len(emitted), 1)
        self.assertAlmostEqual(emitted[0][2], 3.0)
        self.assertAlmostEqual(emitted[0][4], 0.125)


if __name__ == "__main__":
    unittest.main()
