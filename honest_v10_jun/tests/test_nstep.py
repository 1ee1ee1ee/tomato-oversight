from __future__ import annotations

import unittest

import numpy as np

from train import NStepAccumulator


class NStepAccumulatorTests(unittest.TestCase):
    def test_optimized_sum_matches_discounted_definition(self) -> None:
        accumulator = NStepAccumulator(n_steps=3, gamma=0.5)
        emitted = []
        mask = np.asarray([True, True, True, True, False])
        for index, reward in enumerate((1.0, 2.0, 4.0, 8.0)):
            observation = np.asarray([index], dtype=np.float32)
            next_observation = np.asarray([index + 1], dtype=np.float32)
            emitted.extend(
                accumulator.append(
                    (observation, index, reward, next_observation, False, mask)
                )
            )
        self.assertEqual(len(emitted), 2)
        self.assertAlmostEqual(emitted[0][2], 1.0 + 0.5 * 2.0 + 0.25 * 4.0)
        self.assertAlmostEqual(emitted[1][2], 2.0 + 0.5 * 4.0 + 0.25 * 8.0)
        self.assertAlmostEqual(emitted[0][4], 0.5**3)

    def test_terminal_transition_flushes_without_bootstrap(self) -> None:
        accumulator = NStepAccumulator(n_steps=4, gamma=0.9)
        first = np.asarray([0], dtype=np.float32)
        second = np.asarray([1], dtype=np.float32)
        mask = np.asarray([True, True, True, True, False])
        emitted = accumulator.append((first, 0, 1.0, second, False, mask))
        self.assertEqual(emitted, [])
        terminal = np.asarray([2], dtype=np.float32)
        emitted = accumulator.append((second, 1, 2.0, terminal, True, mask))
        self.assertEqual(len(emitted), 2)
        self.assertAlmostEqual(emitted[0][2], 1.0 + 0.9 * 2.0)
        self.assertAlmostEqual(emitted[1][2], 2.0)
        self.assertEqual(emitted[0][4], 0.0)
        self.assertEqual(emitted[1][4], 0.0)


if __name__ == "__main__":
    unittest.main()
