# Ma--Yu 14-Index Gaussian Application

Date: 2026-09-12 Asia/Singapore

Protocol:
`03_data/FROZEN_MA_YU_14_INDEX_GAUSSIAN_PROTOCOL_20260912.md`.

Active immutable run:
`04_experiments/runs/2026-09-12/ma_yu_14_index_gaussian_protocol_v1/`.

## Data

The read-only Ma--Yu canonical matrix contains 1,039 observations of 14 index
returns documented as 2016-01-01 through 2019-12-31. Its verified SHA-256 is
`a6fe95afa29854829c25de140ae820eecc04bb54350048edcc19770a8a638ff4`.
No 2025+ observation and no JW stock-selection outcome were accessed.

## Scalar audit

All 182 directed one-index relations were evaluated in the full, discovery,
and confirmation samples. In every split:

- the maximum closed-form scalar-map error was at most `6.82e-16`;
- GC and canonical CS-TE had zero discordant pairs;
- Spearman correlation and tie-free Kendall approximation were both `1.0`.

This is the required negative control: a scalar target cannot produce the
multivariate ordering counterexample.

## Multivariate block audit

Each split contains 6,006 directed, disjoint 2-index-to-2-index relations and
18,033,015 pairwise ranking comparisons.

| split | discordant pairs | fraction | Spearman |
|---|---:|---:|---:|
| full | 2,651 | 0.0001470 | 0.99999969 |
| discovery | 8,992 | 0.0004986 | 0.99999775 |
| confirmation | 2,992 | 0.0001659 | 0.99999959 |

Thus multivariate ordering differences occur in the real matrix, but they are
rare and global rankings are almost identical.

The discovery-selected strongest reversal was:

- lower-GC edge: `^GSPC+^N225 -> 399001.SZ+^AXJO`, with
  `GC=0.210233`, `CS=0.056540`;
- higher-GC edge: `000001.SS+^STOXX50E -> ^AXJO+^HSI`, with
  `GC=0.210642`, `CS=0.055164`.

On confirmation the first edge had both higher GC and higher CS. The frozen
reversal signs therefore failed, and the 200-replicate moving-block bootstrap
retained both signs in `0%` of replicates.

The frozen near-equal-GC search found a discovery pair separated by only
`5.77e-05` in GC but `0.001508` in CS. This remains exploratory because exact
equality is not expected and no confirmation selection was allowed.

## KDE boundary

For the selected edges, public featurewise KDE scores were approximately
`8.67e-05` and `2.80e-04`, whereas samplewise-median scores were `0.3404` and
`0.5632`. Neither orientation followed the fitted Gaussian magnitudes or their
ordering at fixed `eta=0.05`. This is an implementation sensitivity result, not
a statement about causal truth.

## Decision

**BOUNDARY, not positive financial support.** The data exhibit the permitted
multivariate discordance, but the preselected strongest reversal is temporally
unstable and rankings are nearly identical overall. The financial analysis is
suitable as a limitation or compact case study; it should not be the headline
contribution.
