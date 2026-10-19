# RLO

Lean experiments in learning to solve combinatorial optimization problems, using CVRP as a testbed.

```bash
pip install -e '.[test]'
pytest -q
python -m rlo --n 10 --solver savings --reference exact
```

The core is NumPy-only. `legacy` is the original seed-42 split; `validation` and `test` use separate fixed streams. All routes are independently checked for feasibility and true Euclidean cost. Subset DP is **exact** for up to 12 customers. Larger references are feasible solutions, **not certified optima**.

To generate and freeze route-based references (optional `pip install -e '.[pyvrp]'`):

```bash
python -m scripts.freeze_references --n 20 --split validation --method pyvrp
python -m scripts.freeze_references --n 20 --split test --method pyvrp
```

The original CVRP-20 cost-only archive is kept for historical comparison, **not** as a canonical benchmark: routes were not saved. Use `rlo.evaluate.paired` to compare methods on the instances feasible for both. Old MVP5 checkpoints use the optional PyTorch adapter; they are not stored in Git.
