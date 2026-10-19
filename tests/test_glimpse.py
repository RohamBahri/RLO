"""Guard the optional decoder architecture without changing old checkpoints."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from rlo.data import instances
from rlo.env import CVRPEnv
from rlo.learning import update
from rlo.model import Policy
from rlo.study import model_from_checkpoint
from rlo.train import evaluate_greedy


def test_masked_glimpse_is_permutation_equivariant():
    model = Policy(width=16, layers=1, glimpse=True).eval()
    obs = CVRPEnv(instances(8, "validation", 1)[0]).observe()
    perm = np.array([0, 4, 2, 1, 5, 7, 3, 8, 6])
    with torch.no_grad():
        x, value = model(*(torch.as_tensor(a)[None] for a in obs))
        changed = (obs[0][perm], obs[1], obs[2][np.ix_(perm, perm)], obs[3][perm])
        xp, vp = model(*(torch.as_tensor(a)[None] for a in changed))
    np.testing.assert_allclose(xp.numpy(), x[:, perm].numpy(), atol=2e-5)
    assert vp.item() == pytest.approx(value.item(), abs=2e-6)


def test_glimpse_backprop_and_multistart_validation():
    model = Policy(width=16, layers=1, glimpse=True)
    before = model.glimpse_q.weight.detach().clone()
    torch.manual_seed(3)
    update(model, torch.optim.Adam(model.parameters(), lr=.01),
           instances(8, "validation", 8), "a2c")
    assert not torch.equal(before, model.glimpse_q.weight.detach())
    result = evaluate_greedy(model, [6], 3, multistart=True)["6"]
    assert result["multistart_cost"] <= result["cost"] + 1e-8


def test_v3_checkpoint_roundtrip(tmp_path):
    original = Policy(width=16, layers=1, glimpse=True)
    checkpoint = tmp_path / "model.pt"
    arch = dict(width=16, layers=1, glimpse=True, logit_clip=10., distance_bias=True)
    torch.save({"architecture": arch, "architecture_version": 3,
                "model": original.state_dict()}, checkpoint)
    restored = model_from_checkpoint(checkpoint, device="cpu")
    assert restored.glimpse
    assert all(torch.equal(a, b) for a, b in zip(original.parameters(), restored.parameters()))
