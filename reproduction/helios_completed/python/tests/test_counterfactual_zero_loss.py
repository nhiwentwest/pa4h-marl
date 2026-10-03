import unittest
import torch
from marl_gang_train import counterfactual_policy_loss


class CounterfactualZeroLossTests(unittest.TestCase):
    def test_tied_legal_actions_have_zero_finite_loss_and_gradients(self):
        raw = torch.tensor([[0.2, -0.1, 0.8]], requires_grad=True)
        masks = torch.tensor([[True, False, True]])
        utilities = torch.tensor([[0.5, -torch.inf, 0.5]])
        loss, rows = counterfactual_policy_loss(raw.masked_fill(~masks, -torch.inf), utilities, masks)
        self.assertEqual(rows, 0)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.equal(raw.grad, torch.zeros_like(raw)))

    def test_forced_and_no_legal_rows_have_zero_finite_loss(self):
        for masks in (torch.tensor([[True, False, False]]), torch.tensor([[False, False, False]])):
            raw = torch.tensor([[0.2, -0.1, 0.8]], requires_grad=True)
            utilities = torch.tensor([[0.5, -torch.inf, -torch.inf]])
            loss, rows = counterfactual_policy_loss(raw.masked_fill(~masks, -torch.inf), utilities, masks)
            self.assertEqual(rows, 0)
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            self.assertTrue(torch.equal(raw.grad, torch.zeros_like(raw)))


if __name__ == '__main__':
    unittest.main()
