import numpy as np
import torch

from a4_policy_regret import policy_regret_loss


def value_and_gradient(logits, values, mask=None):
    x = torch.tensor([logits], dtype=torch.float64, requires_grad=True)
    m = torch.ones_like(x, dtype=torch.bool) if mask is None else torch.tensor([mask])
    loss, info = policy_regret_loss(x, m, torch.tensor([values], dtype=torch.float64))
    loss.backward()
    return loss.detach(), x.grad[0], info


def test_each_tied_best_action_receives_supervision():
    _, gradient, info = value_and_gradient([.1, .2, .4, -2.], [2., 2., 0., np.nan])
    assert info['active'] == 1
    assert bool((gradient[:2] < 0).all())
    assert gradient[2] > 0 and gradient[3] == 0


def test_anchor_numbering_does_not_choose_a_different_teacher():
    logits = np.array([.1, .2, .4, -2.])
    values = np.array([2., 2., 0., np.nan])
    permutation = np.array([1, 0, 2, 3])
    loss, grad, _ = value_and_gradient(logits.tolist(), values.tolist())
    moved_loss, moved_grad, _ = value_and_gradient(logits[permutation].tolist(), values[permutation].tolist())
    torch.testing.assert_close(loss, moved_loss)
    torch.testing.assert_close(grad[permutation], moved_grad)


def test_ties_use_the_same_epsilon_as_regret_detection():
    _, gradient, _ = value_and_gradient([.1, .2, .4, -2.], [2., 2.+1e-8, 0., np.nan])
    assert bool((gradient[:2] < 0).all())


def test_unknown_and_masked_actions_do_not_receive_labels():
    loss, gradient, _ = value_and_gradient(
        [.1, .2, .4, .1, -np.inf, -2.], [2., 2., 0., np.nan, np.nan, np.nan],
        [True, True, True, True, False, True])
    assert torch.isfinite(loss) and bool(torch.isfinite(gradient).all())
    assert torch.equal(gradient[3:], torch.zeros_like(gradient[3:]))


def test_already_best_or_entirely_tied_states_have_no_routing_gradient():
    for values in ([2., 2., 0., np.nan], [2., 2., 2., np.nan]):
        loss, gradient, info = value_and_gradient([.4, .2, .1, -2.], values)
        assert loss == 0 and info['active'] == 0
        assert torch.equal(gradient, torch.zeros_like(gradient))
