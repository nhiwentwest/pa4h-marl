import inspect
import unittest
import torch
import torch.nn.functional as F
from models import RackSTGNNActor


def actor():
    # Run the same behavioral regression against the old and repaired classes.
    kwargs = dict(context_dim=4, include_defer=True, hidden=8, heads=2)
    if 'action_context_dim' in inspect.signature(RackSTGNNActor).parameters:
        kwargs['action_context_dim'] = 3
    return RackSTGNNActor(2, 2, 1, **kwargs)


class ActionConditionedRackTests(unittest.TestCase):
    def test_can_learn_opposite_racks_for_identical_telemetry(self):
        torch.manual_seed(4)
        model = actor()
        # Same telemetry; two jobs have opposite phase-aligned anchor utilities.
        x = torch.tensor([[0.,0.,0.,0., 0.,1.,-1.,-2.],
                          [0.,0.,0.,0., 1.,-1.,1.,-2.]])
        mask = torch.tensor([[True,True,False], [True,True,False]])
        targets = torch.tensor([0,1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        for _ in range(80):
            loss = F.cross_entropy(model(x, mask), targets)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        logits = model(x, mask)
        self.assertEqual(logits.argmax(-1).tolist(), [0,1])
        self.assertLess(F.cross_entropy(logits, targets).item(), 0.05)

    def test_job_context_can_change_relative_rack_scores(self):
        torch.manual_seed(7)
        model = actor()
        x = torch.randn(1, model.obs_dim, requires_grad=True)
        mask = torch.tensor([[True,True,False]])
        probabilities = model(x, mask).softmax(-1)
        probabilities[0,0].backward()
        self.assertGreater(abs(x.grad[0,model.graph_dim].item()), 1e-8)

    def test_masked_action_has_zero_probability_and_finite_gradients(self):
        torch.manual_seed(7)
        model = actor()
        x = torch.randn(2, model.obs_dim)
        mask = torch.tensor([[True,False,True], [False,True,False]])
        logits = model(x,mask)
        loss = F.cross_entropy(logits, torch.tensor([0,1]))
        loss.backward()
        self.assertTrue(torch.equal(logits.softmax(-1)[~mask], torch.zeros(3)))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all().item()
                            for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
