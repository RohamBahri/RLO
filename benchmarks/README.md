# References

`legacy_n20.json` contains the **original** archived OR-Tools rounded cost values only, not tours. The original tours were never saved. **Do not treat these values as validated Euclidean costs.** Additional archived means are 6.373003 and 6.358619, but none of the underlying tours can be reconstructed from the logged costs alone.

Generate validated route-based references using `python -m scripts.freeze_references`. Small n≤12 instances have exact subset-DP references; larger instances can use multiple seeded PyVRP runs. On reload, instance fingerprints, every tour, and its stored Euclidean cost are all rechecked. References for n≥20 validation/test are not yet generated.
