"""Regression checks for fair Group A comparisons and cached attention."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from rlo.data import instances, training_batches
from rlo.env import CVRPEnv
from rlo.learning import rollout, update
from rlo.model import Policy
from rlo.study import best_of_k


def test_encoder_runs_once_per_batch_and_receives_gradients():
    model = Policy(width=16, layers=1)
    count = []
    hook = model.encoder[0].register_forward_hook(lambda *args: count.append(1))
    before = model.embed.weight.detach().clone()
    cases = instances(8, "validation", 4)
    update(model, torch.optim.Adam(model.parameters(), lr=1e-3), cases, "a2c")
    hook.remove()
    assert len(count) == 1
    assert not torch.equal(before, model.embed.weight.detach())


def test_size_sensor_and_clipped_logits():
    for n in (4, 20):
        env = CVRPEnv(instances(n, "test", 1)[0])
        node, global_features, distance, mask = env.observe()
        assert global_features[1] == pytest.approx(1 / n)
        model = Policy(width=16, layers=1)
        with torch.no_grad():
            logits, _ = model(*[torch.as_tensor(x).unsqueeze(0)
                                for x in (node, global_features, distance, mask)])
        assert logits[0][torch.as_tensor(mask)].abs().max() <= 10.00001


def test_full_pomo_and_frozen_baseline():
    cases = instances(6, "validation", 4)
    model = Policy(width=16, layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    frozen = copy.deepcopy(model)
    frozen_weights = {k: v.detach().clone() for k, v in frozen.state_dict().items()}
    record = update(model, opt, cases, "rollout", baseline_policy=frozen)
    assert record["baseline_episodes"] == 4
    assert all(torch.equal(frozen.state_dict()[k], v) for k, v in frozen_weights.items())
    batches_seen = []
    hook = model.encoder[0].register_forward_hook(lambda layer, args, out: batches_seen.append(out.shape[0]))
    record = update(model, opt, cases, "pomo", rng=np.random.default_rng(2))
    hook.remove()
    assert batches_seen == [4]  # attention encodes only four unique instances, not 24 starts
    assert record["starts"] == 6
    assert record["episodes"] == 24


def test_training_batches_are_same_size_and_reproducible():
    a = training_batches(9, batch=4, sizes=(6, 10), capacity_factors=(.5, 1.5))
    b = training_batches(9, batch=4, sizes=(6, 10), capacity_factors=(.5, 1.5))
    for _ in range(5):
        x, y = next(a), next(b)
        assert len({len(inst["demands"]) for inst in x}) == 1
        assert all(np.array_equal(i["coords"], j["coords"]) for i, j in zip(x, y))


def test_multistart_inference_generates_valid_solutions():
    model = Policy(width=16, layers=1)
    cases = instances(6, "test", 2)
    solutions, steps = best_of_k(model, cases, k=6, multistart=True, return_transitions=True)
    assert len(solutions) == 2 and all(item is not None for item in solutions)
    assert steps >= 2 * 6 * 6
    with pytest.raises(ValueError, match="cannot exceed"):
        best_of_k(model, cases, k=7, multistart=True)


def test_training_transition_cap_and_new_checkpoint(tmp_path):
    folder = tmp_path / "test"
    subprocess.run([sys.executable, "-m", "rlo.train", "--sizes", "6", "8",
                    "--width", "16", "--layers", "1", "--batch", "2",
                    "--updates", "100", "--max-transitions", "40",
                    "--validate-every", "100", "--validation-count", "2",
                    "--out", str(folder)], check=True, capture_output=True, text=True)
    records = [json.loads(s) for s in (folder / "history.jsonl").read_text().splitlines()]
    assert 40 <= records[-1]["transitions_total"]
    assert len(records) < 100
    ckpt = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
    assert ckpt["architecture"]["logit_clip"] == 10.0
