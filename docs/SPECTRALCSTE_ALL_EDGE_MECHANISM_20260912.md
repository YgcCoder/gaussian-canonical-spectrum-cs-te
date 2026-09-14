# SpectralCSTE All-Edge Mechanism Test

Date: 2026-09-12 Asia/Singapore

Decision: **MECHANISM NOT CONFIRMED AS STABLE FINANCIAL EVIDENCE**.

Protocol was frozen before calculation in
`03_data/FROZEN_SPECTRALCSTE_ALL_EDGE_MECHANISM_PROTOCOL_20260912.md`.
The active immutable run is
`04_experiments/runs/2026-09-12/spectralcste_all_edge_mechanism_v1/`.

## Theory-fixed mechanism

For two canonical transfer components `x1,x2`, define

\[
H=\frac{4x_1x_2}{(x_1+x_2)^2}
\]

as spectrum spread and

\[
Q=\phi(x_1+x_2)-\phi(x_1)-\phi(x_2)
\]

as the loss relative to the rank-one CS score at the same GC. The normalized
gap is

\[
Q_N=\frac{Q}{\phi(x_1+x_2)-2\phi((x_1+x_2)/2)}.
\]

Both lie in `[0,1]`: zero denotes rank-one concentration and one denotes equal
two-mode spreading. The theory fixes the direction `larger H -> larger Q`.

## Discovery-only thresholds

All 6,006 discovery edges were used. No edge identifier entered threshold
selection.

- Near-GC tolerance: `tau=0.0003064706`, equal to one percent of median
  positive discovery GC.
- Dispersion threshold: `kappa=0.2602902`, the median nonzero `|Delta H|`
  among all 118,301 discovery near-GC comparisons.
- Qualified discovery comparisons: 59,151.

## Algebraic mechanism check

| split | Pearson `H,Q_N` | Spearman `H,Q_N` | mean `H` | mean raw `Q` |
|---|---:|---:|---:|---:|
| discovery | 0.99999999997 | 0.99999999784 | 0.26627 | 3.94e-05 |
| confirmation | 0.99999999998 | 0.99999999795 | 0.19483 | 2.52e-05 |

This near-identity is an internal spectral-algebra verification, not independent
financial evidence: both variables are deterministic functions of the same two
canonical components.

Maximum GC-versus-spectrum errors were `9.04e-15` in discovery and `6.36e-15`
in confirmation.

## Confirmation all-edge ordering test

Applying the frozen thresholds to confirmation produced 170,689 near-GC
comparisons and 81,606 dispersion-qualified comparisons.

| quantity | discovery | confirmation |
|---|---:|---:|
| concentration-rule directional accuracy | 0.5894 | 0.5168 |
| predicted-reversal comparisons | 29,480 | 40,650 |
| reversal fraction in predicted group | 0.1761 | 0.02989 |
| reversal fraction in opposite group | 0 | 0 |
| reversal risk difference | 0.1761 | 0.02989 |
| predicted-group mean signed CS gap | -2.23e-05 | -3.55e-05 |

Although dispersion enriches reversal frequency relative to the opposite group,
the confirmation directional accuracy is only 1.68 percentage points above
chance and the mean signed CS difference remains negative. The lower-GC edge
usually does not overcome even the small GC gap.

## Temporal stability and bootstrap

The entire discovery-qualified pair set was frozen; no edge was selected.

- Confirmation retention of the discovery-predicted CS sign: `0.5291`.
- Discovery-observed reversal pairs: 5,192.
- Pairs retaining both original GC and CS signs on confirmation: 1 of 5,192,
  or `0.0001926`.

All 200 circular moving-block bootstrap replicates completed. Percentile 95%
intervals were:

| statistic | interval |
|---|---:|
| standardized `H -> Q_N` slope | `[0.99999999992, 0.99999999999]` |
| cross-sectional directional accuracy | `[0.5293, 0.6518]` |
| predicted-reversal fraction | `[0.0566, 0.3031]` |
| reversal risk difference | `[0.0566, 0.3031]` |
| frozen-pair predicted-sign retention | `[0.4840, 0.5453]` |
| frozen observed-reversal retention | `[0, 0.000963]` |

The bootstrap distribution of cross-sectional accuracy is upward shifted from
the unresampled confirmation estimate, consistent with resampling sensitivity.
The directly relevant temporal interval includes chance, while observed
reversal retention is essentially zero.

## Scalar sanity check

All 182 scalar edges again had zero GC-versus-CS ranking discordances. Maximum
closed-form map errors were `5.13e-16` in discovery and `4.79e-16` in
confirmation.

## Publication route

The frozen decision rule is not met because temporal stability is negligible.
Therefore:

1. keep the exact theory and controlled synthetic evidence in the four-page
   main paper;
2. place the 14-index all-edge analysis only in an appendix, limitations, or a
   compact negative-boundary paragraph;
3. do not select or foreground any individual financial edge;
4. do not claim a stable market-ranking benefit for SpectralCSTE.
