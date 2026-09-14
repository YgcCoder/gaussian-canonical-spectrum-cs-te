# Gaussian Canonical-Spectrum Geometry of CS-TE

This local release is the code-and-results package intended for a future GitHub
repository. The Overleaf archive contains only the paper source and figure; it
does not contain experiment code or data.

## Contents

- `scripts/`: population, stress, finite-sample, and financial-audit programs;
- `configs/`: frozen JSON configurations;
- `data/`: the sealed 14-index return matrix used by the auxiliary audit;
- `results/`: immutable outputs reported or discussed in the paper;
- `docs/`: protocol and interpretation reports;
- `supplementary/financial_audit/README.md`: exact data, preprocessing, index,
  metric, and limitation definitions.

All experiment scripts refuse to overwrite an existing output directory.

## Environment

The release was verified with Python 3 and NumPy 2.0.2. Install the sole
third-party dependency with:

```bash
python3 -m pip install -r requirements.txt
```

Run commands from the repository root.

## Main paper experiments

Population construction audit:

```bash
PYTHONPATH=scripts python3 scripts/run_gaussian_cs_te_equivalence.py \
  --config configs/gaussian_cs_te_equivalence_v1.json \
  --output reproduction/gaussian_cs_te_equivalence_v1
```

Conditioning and KDE stress audit:

```bash
PYTHONPATH=scripts python3 scripts/run_gaussian_cs_te_stress_v2.py \
  --config configs/gaussian_cs_te_stress_v2.json \
  --output reproduction/gaussian_cs_te_stress_v2
```

Finite-sample reversal recovery:

```bash
PYTHONPATH=scripts python3 scripts/run_finite_sample_reversal_stability.py \
  --config configs/finite_sample_reversal_stability_v1.json \
  --output reproduction/finite_sample_reversal_stability_v1
```

The committed finite-sample outputs were independently rerun and reproduced
byte for byte. See `docs/FINITE_SAMPLE_REVERSAL_STABILITY_RESULTS_20260914.md`.

## Auxiliary financial boundary audit

```bash
PYTHONPATH=scripts python3 scripts/run_ma_yu_14_index_gaussian_protocol.py \
  --config configs/ma_yu_14_index_gaussian_protocol_v1.json \
  --output reproduction/ma_yu_14_index_gaussian_protocol_v1

PYTHONPATH=scripts python3 scripts/run_spectralcste_all_edge_mechanism.py \
  --config configs/spectralcste_all_edge_mechanism_v1.json \
  --output reproduction/spectralcste_all_edge_mechanism_v1
```

The data file is hash-checked by both programs before use. The financial audit
is deliberately reported as a negative boundary: multivariate ranking
discordances are rare and temporally unstable.

## Reproducible outputs

- population examples: `(G,T_CS)` equal to `(2,0.7564417556)`,
  `(2,0.6559778785)`, and `(2.2,0.7350795174)`;
- finite-sample joint reversal recovery: `0.226` at `n=64` and `0.875` at
  `n=4096` over 1,000 repetitions per sample size;
- auxiliary finite-sample KDE diagnostics across three bandwidths (these
  orientation- and bandwidth-sensitive correlations are not consistency or
  performance results):
  Spearman values are
  feature-column `[-0.471,-0.524,-0.458]`, sample-row
  `[-0.076,-0.605,-0.528]`;
- financial confirmation directional accuracy: `0.5168`, where the metric is
  within-split sign agreement, not market prediction accuracy.
