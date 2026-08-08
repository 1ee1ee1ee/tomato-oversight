from __future__ import annotations

import unittest

import torch

from src.ddqn import masked_argmax


class ActionMaskingTests(unittest.TestCase):
    def test_masked_argmax_ignores_invalid_maximum(self) -> None:
        q_values = torch.tensor([[1.0, 100.0, 3.0]])
        mask = torch.tensor([[True, False, True]])
        self.assertEqual(int(masked_argmax(q_values, mask).item()), 2)


if __name__ == "__main__":
    unittest.main()
