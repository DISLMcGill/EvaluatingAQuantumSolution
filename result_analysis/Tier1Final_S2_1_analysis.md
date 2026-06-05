# Critical analysis — `Tier1Final_S2_1.json`

Source: `result_bank/quantum_hardware_results/Tier1Final_S2_1.json`
Run: 2026-06-01, `tier1_subset_sqa_hw.py`, ILP baseline + `SQA_SF_HW` (S2) on `Advantage_system4` (Pegasus).
Stated config: `annealing_time=50µs`, `num_reads=500/1000`, `chain_strength = uniform_torque_compensation(prefactor=1.5)` flat.

---

## 1. The run is not what the file claims it is

The metadata advertises a Tier-1 sweep of **36 cases** (12 `(n,p)` configs × 3 tightness levels `t30/t70/t100`). The `results` dict contains **12 entries**, *all at `tc_tightness=0.7`*. This is not a partial-run truncation: every `(n,p)` combination appears exactly once, and only the `t70` instance survived.

Root cause is in `util/experiment_execution/run_experiment.py:436`:

```python
key = tc_path.stem        # e.g. "n-3_p-12_1"
...
output["results"][key] = entry
```

`tc_path` walks `tier1/n3_p12/{t30,t70,t100}/n-3_p-12_1.json`. The tightness folder is *not* part of `tc_path.stem`, so all three tightness variants of the same `(n,p)` collide on a single dict key. The harness submits all 36 jobs to the QPU, then silently overwrites results on each subsequent insert. Lex order on directory names is `t100 < t30 < t70`, so `t70` is processed last and is the only tightness preserved. **Two-thirds of the QPU data charged to this run is gone.**

This invalidates the headline framing of the file. Any tightness-sensitivity claim built on top of it is unsupported; what's here is a single tightness slice, not a Tier-1 sweep.

## 2. Solver-quality summary across the 10 cases that produced output

| case        | n | p  | ILP cost | SQA cost | rel. gap | valid | k-viol | cap-over | cbf    | best-occ | reads | phys-q | max-chain |
|-------------|---|----|---------:|---------:|---------:|:-----:|-------:|---------:|-------:|---------:|------:|-------:|----------:|
| n-3_p-4     | 3 |  4 |       29 |       29 |     0.0% |   ✓   |      0 |        0 | 0.0000 |        6 |    87 |     18 |         2 |
| n-3_p-12    | 3 | 12 |      168 |      312 |    85.7% |   ✓   |      0 |        0 | 0.0003 |        1 |   500 |    106 |         5 |
| n-3_p-26    | 3 | 26 |      176 |      493 |   180.1% |   ✓   |      0 |        0 | 0.0004 |        1 |  1000 |    462 |        10 |
| n-3_p-50    | 3 | 50 |      859 |    1 405 |    63.6% |   ✓   |      0 |        0 | 0.0026 |        1 |  1000 |  1 614 |        18 |
| n-5_p-4     | 5 |  4 |      296 |      317 |     7.1% |   ✓   |      0 |        0 | 0.0001 |        1 |   830 |     41 |         3 |
| n-5_p-12    | 5 | 12 |      665 |      874 |    31.4% |   ✓   |      0 |        0 | 0.0007 |        1 |  1000 |    252 |         6 |
| n-5_p-26    | 5 | 26 |    1 352 |    2 819 |   108.5% |   ✓   |      0 |        0 | 0.0073 |        1 |  1000 |  1 353 |        19 |
| n-5_p-50    | 5 | 50 |    2 949 |      —   |        — |  ✗ embed-fail | — | — |      — |        — |     — |      — |         — |
| n-9_p-4     | 9 |  4 |      658 |      664 |     0.9% |   ✓   |      0 |        0 | 0.0026 |        1 |   995 |    106 |         4 |
| n-9_p-12    | 9 | 12 |    1 104 |    1 548 |    40.2% |   ✗   |      0 |        2 | 0.0503 |        1 |  1000 |    909 |        14 |
| n-9_p-26    | 9 | 26 |    4 736 |    6 280 |    32.6% |   ✗   |     15 |        1 | 0.1502 |        1 |  1000 |  4 012 |        33 |
| n-9_p-50    | 9 | 50 |    8 572 |      —   |        — |  ✗ embed-fail | — | — |      — |        — |     — |      — |         — |

Aggregates over the 10 cases that ran:

- **Match to ILP optimum**: 1/10 (only the trivial 12-variable `n-3_p-4`).
- **Valid but suboptimal**: 7/10. Median relative gap **36.4%**, max **180%**.
- **Invalid (constraint-violating output)**: 2/10. Both at `n=9` with the larger BQMs.
- **Embedding failure**: 2/12. Both at `p=50` for `n∈{5,9}` — the heuristic minor-embedder couldn't lay out 250+ logical vars on Pegasus.

## 3. Convergence has not been achieved

`best_num_occurrences = 1` on **9 of 10** cases that ran. The lone exception is the 12-variable `n-3_p-4`. In other words: even when the QPU stumbled into the ground state, it did so once in 500–1000 reads. The metadata text in the file acknowledges this is exactly the symptom that motivated bumping anneal time and doubling reads from the prior tuning runs — *and the symptom persists in the locked-in config*. The supposed remedy did not work.

The cost of an under-sampled SQA run is invisible: with 1-out-of-1000 occurrence, there is no statistical evidence the reported energy is the true ground state of the BQM, never mind the original objective. The 180% gap on `n-3_p-26` is the natural reading of "we sampled a non-ground-state and called it done."

## 4. The "locked" hyperparameters don't actually hold

`schedule_num_reads` returns `500` or `1000`. The result file reports:

- `n-3_p-4`: **87** reads (schedule says 500).
- `n-5_p-4`: **830** reads (schedule says 1000).
- `n-9_p-4`: **995** reads (schedule says 1000).

The QPU's `problem_run_duration_range` caps total annealing+readout time per submission, and `EmbeddingComposite` silently truncates `num_reads` when that ceiling is hit. The harness records the *requested* schedule in metadata but the *actual* `num_reads` per submission on each entry — and they disagree by up to **6×** on the smallest case. So the "convergence problem" diagnosis in §3 may be partly an artefact of the harness asking for 500 reads and the chip delivering 87, with no flag raised.

This is a real instrumentation gap. The convergence story would need to be redone with verified `num_reads` actually executed.

## 5. Chain-strength prefactor 1.5 does not generalize beyond `n=3`

The tuning history embedded in metadata explains how the team arrived at `prefactor=1.5` flat by lowering it run-by-run on the `n=3` instances, observing chain-break fraction stayed "50–500× below the 0.05 concern threshold" on those cases. They then locked the value and ran the full sweep.

What the locked sweep actually shows:

- `n-3_p-{4..50}`: cbf 0–0.0026. Fine.
- `n-5_p-{4..26}`: cbf 0.0001–0.0073. Fine.
- **`n-9_p-12`: cbf 0.0503**. Above the team's own concern threshold. Invalid solution.
- **`n-9_p-26`: cbf 0.1502**. 3× the threshold. 15 k-safety violations, 1 capacity overrun.

The hypothesis ("1.5 is enough chain coupling") was validated on a problem class that turned out not to be representative. As soon as embeddings reach mean chain length ≈ 8 (n9_p12) and 17 (n9_p26), chains start breaking, the majority-vote unembedding mangles the logical solution, and the result is infeasible. The script's own escape clause ("if cbf rises notably on n3_p50, 1.5 is the floor and we revert to 1.5/2.0") was monitoring the wrong case — n3 was never going to be the bottleneck.

## 6. Lambda calibration mixes two incompatible regimes within one sweep

`SQA_SF.calibrate_lambdas` switches modes at `n_vars ≤ 18`:

- **Exact path** (n_vars ≤ 18): grid-search `(λ₁, λ₂)` against `dimod.ExactSolver`. Used here only by `n-3_p-4` (12 vars) → λ₁/λ₂ = 394.5 / 0.1, ratio **3945**.
- **Heuristic path** (n_vars > 18): closed form `λ₂ = h/max_C`, `λ₁ = 2·max_C·λ₂`. Used for every other case → ratios 2–84.

The only case that matched ILP cost is the only case that used the exact path. Every heuristic-lambda case produced a strictly inferior result. This is at least suggestive that the heuristic fallback is mis-calibrated, *not* that the QPU is failing — the QPU's BQM has the wrong ground state to find. Distinguishing these explanations requires either (a) decoding the best raw sample back to the original objective and checking whether the BQM's ground state IS the ILP optimum, or (b) running the simulated S2 on the same BQM at full sweep budget and comparing. Neither is done here.

The `lambda_2 = 0.1` value for `n-3_p-4` is itself worth flagging: the exact grid sweeps `l2 ∈ {0.1, 0.25, 0.5, 1.0, ...}` and picks whichever feasible pair minimises the gap to ILP. Picking the *smallest* l2 means the penalty parabola is extremely shallow; this works when the exact solver verifies a feasible ground state exists, but it would be a catastrophic choice on hardware where finite-temperature noise needs a stiffer penalty. The pattern is invisible in this sweep because the only "exact" case is also the smallest, but it's a footgun if `max_vars_for_exact` is ever raised.

## 7. Runtime comparison is not informative

Reported `wall_time_ms` (the SQA column) sums to ~231 s across 10 cases. Pure `qpu_access_time_us` sums to ~2.3 s. The remaining 99% is queue + network. ILP wall time sums to 1.16 s on a local CBC. So:

- If you compare wall-clock end-to-end: ILP is ~200× faster *and* better.
- If you compare QPU-access only: ILP is ~2× faster *and* better.
- Neither comparison favors the QPU on this problem class at this size.

Worse, ILP solved every case (no embedding failures, no invalid outputs) and produced a proven-optimal cost (`optimality_gap_absolute=0`). There is no metric in the dataset where the hardware run beats the classical baseline.

## 8. Bottom-line assessment

The file labels itself "final locked config" for a 36-case Tier-1 sweep. In practice it is:

1. A 12-case slice (silent overwrite bug; 24 QPU submissions discarded).
2. With 2 embedding failures (17%).
3. With 2 invalid outputs (20% of the cases that did embed).
4. With a 7/8 ratio of valid-but-suboptimal results (median 36% above ILP, max 180%).
5. Where the only ILP-matching case is the only one whose lambdas were exactly calibrated, and the only one with `best_num_occurrences > 1`.
6. With actual `num_reads` ≠ scheduled `num_reads` on at least three cases, undetected by the harness.
7. With chain-break behaviour above the team's own concern threshold on every `n=9` case that produced output.

This is not a result that supports any claim of quantum advantage, or even of competitive performance, for the S2 formulation on `Advantage_system4` at Tier-1 sizes. It is a result that shows the pipeline runs end-to-end and that the BQM construction is correct enough to be feasible on small instances. Any subsequent comparison needs to (a) fix the dict-key collision, (b) record actual executed `num_reads`, (c) re-tune `chain_strength` on the n=9 cases that drove the failures, and (d) decouple "BQM ground state ≠ ILP optimum" from "QPU didn't find the BQM ground state" before attributing the gaps to hardware.
