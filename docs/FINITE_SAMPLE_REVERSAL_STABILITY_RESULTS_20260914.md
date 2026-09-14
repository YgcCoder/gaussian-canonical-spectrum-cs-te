# Finite-Sample Reversal Stability Results

Date: 2026-09-14 Asia/Singapore

Protocol: `04_experiments/reports/FINITE_SAMPLE_REVERSAL_STABILITY_PLAN_20260914.md`

Config: `04_experiments/configs/finite_sample_reversal_stability_v1.json`

Immutable run: `04_experiments/runs/2026-09-14/finite_sample_reversal_stability_v1/`

## Question

The population theory gives three feasible nilpotent Gaussian VAR(1) models:

- `A=(1,0)`: `G=2.0`, `T_CS=0.7564417556`;
- `B=(0.5,0.5)`: `G=2.0`, `T_CS=0.6559778785`;
- `C=(0.55,0.55)`: `G=2.2`, `T_CS=0.7350795174`.

The experiment asks when covariance estimates recover (i) the equal-GC
separation `T_A>T_B` and (ii) the joint reversal `G_A<G_C` and `T_A>T_C`.

## Design

- sample sizes: 64, 128, 256, 512, 1024, 2048, 4096;
- 1,000 independent replicate triples per sample size;
- master seed: `20260914`;
- burn-in: 64;
- Gaussian sample-covariance plug-in;
- Wilson 95% intervals;
- 7,000 completed replicate triples and zero failures.

## Results

| n | Equal-GC separation | Joint reversal |
|---:|---:|---:|
| 64 | 0.739 [0.711, 0.765] | 0.226 [0.201, 0.253] |
| 128 | 0.821 [0.796, 0.844] | 0.338 [0.309, 0.368] |
| 256 | 0.913 [0.894, 0.929] | 0.470 [0.439, 0.501] |
| 512 | 0.974 [0.962, 0.982] | 0.592 [0.561, 0.622] |
| 1024 | 1.000 [0.996, 1.000] | 0.719 [0.690, 0.746] |
| 2048 | 1.000 [0.996, 1.000] | 0.781 [0.754, 0.806] |
| 4096 | 1.000 [0.996, 1.000] | 0.875 [0.853, 0.894] |

The smaller population CS-TE margin for the reversal (`0.0213`) requires more
data than the equal-GC contrast (`0.1005`). The result supports finite-sample
visibility of two fixed population facts; it is not a general power theorem.

## Reproduction check

A fresh rerun in `/tmp/finite_recheck.ix5inP/recheck` produced byte-identical
`results.json`, `recovery_by_sample_size.csv`, and `replicate_metrics.csv`.

- config SHA-256: `29c014d08558cd7e6efdd2937bb76fe697b9316d2de2ec1563ba6af0b9350bfe`;
- results SHA-256: `65e855173b7d126cf7e258fd65eba66c0d96a70b95c11cbeb87ce87284eacd67`;
- summary SHA-256: `eb31dd3c56025e2f10e87f3df3ff6f7d80d4da17ddad2059f4a730eb08f3ed80`;
- replicate metrics SHA-256: `0a532471eba488df6c2b2215d63061dc579d5f05b6379b69a0547fc402fa258e`.

