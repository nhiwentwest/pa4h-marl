"""Regression tests for PPO entropy with action-masked logits."""
import unittest

import torch
from torch.distributions import Categorical

from marl_gang_train import (counterfactual_policy_loss,
                             masked_categorical_entropy)
from models import RackSTGNNActor


class MaskedPolicyEntropyTests(unittest.TestCase):
    def test_masked_entropy_matches_categorical_and_has_finite_gradients(self):
        logits = torch.tensor(
            [[-0.8, 0.3, -0.07950062, 1.1, -0.4, 2.1108432]],
            dtype=torch.float32,
            requires_grad=True,
        )
        mask = torch.tensor([[False, False, True, False, False, True]])
        masked_logits = logits.masked_fill(~mask, float("-inf"))

        entropy = masked_categorical_entropy(masked_logits, mask)
        reference = Categorical(logits=masked_logits).entropy()

        self.assertTrue(torch.allclose(entropy, reference, atol=1e-7, rtol=1e-7))
        entropy.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertTrue(torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask])))

    def test_masked_entropy_and_counterfactual_loss_backpropagate_through_stgnn(self):
        actor = RackSTGNNActor(
            num_racks=4, history_len=2, feat_dim=3, context_dim=4,
            hidden=8, heads=2,
        )
        obs = torch.randn(1, 28)
        mask = torch.tensor([[False, True, False, True]])
        utilities = torch.tensor([[float("-inf"), 0.2, float("-inf"), 0.8]])
        logits = actor(obs, mask)
        entropy = masked_categorical_entropy(logits, mask).mean()
        cf_loss, rows = counterfactual_policy_loss(logits, utilities, mask)

        (entropy + cf_loss).backward()

        self.assertEqual(rows, 1)
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in actor.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
