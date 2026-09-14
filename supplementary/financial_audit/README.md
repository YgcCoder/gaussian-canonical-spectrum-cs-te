# Auxiliary 14-Index Financial Boundary Audit

This audit is retained for reproducibility and boundary analysis. It is not a
headline experiment and does not support a claim that CS-TE ranks markets
better than GC.

## Data provenance

The read-only source is the return matrix released with the public Ma--Yu
CS-TE notebook. The notebook downloads daily closing prices from 2016-01-01 to
2019-12-31, forward-fills missing prices, and forms returns. The canonical
matrix has 1,039 rows and 14 columns and does not retain dates.

SHA-256:
`a6fe95afa29854829c25de140ae820eecc04bb54350048edcc19770a8a638ff4`

The 14 columns are:

| Symbol | Index |
|---|---|
| `000001.SS` | Shanghai Composite |
| `399001.SZ` | Shenzhen Component |
| `IMOEX.ME` | MOEX Russia |
| `^AXJO` | S&P/ASX 200 |
| `^DJI` | Dow Jones Industrial Average |
| `^GSPC` | S&P 500 |
| `^HSI` | Hang Seng |
| `^IXIC` | Nasdaq Composite |
| `^KS11` | KOSPI |
| `^N100` | Euronext 100 |
| `^N225` | Nikkei 225 |
| `^NZ50` | S&P/NZX 50 |
| `^STI` | Straits Times |
| `^STOXX50E` | EURO STOXX 50 |

Because exchanges have asynchronous calendars, forward filling can generate
zero returns on local holidays. The matrix is used exactly as stored: no
refetching, winsorization, standardization, or tuned transformation.

## Fixed protocol

- lag: one trading-row lag;
- model: intercept-containing restricted and unrestricted OLS regressions;
- residual covariance: maximum-likelihood normalization;
- discovery rows: `[0,520)`;
- confirmation rows: `[520,1039)`;
- scalar audit: all 182 ordered one-to-one relations;
- multivariate audit: all 6,006 ordered, disjoint two-index-source to
  two-index-target blocks;
- moving-block bootstrap: 200 circular replicates with block length 20.

## What “directional accuracy” means

For every near-GC pair, first order the two edges as low-GC and high-GC. Let
`H` denote spectrum spread. The theory-derived rule predicts the sign of
`CS_low - CS_high` from the sign of `H_high - H_low`. Directional accuracy is
the fraction of those predicted signs that match the observed Gaussian
plug-in CS-TE gap on the same split.

It is not accuracy for market direction, returns, a causal ground-truth label,
or an externally observed outcome. The confirmation value `0.5168` is near
chance and must not be described as predictive performance.

## Boundary result

- discovery/confirmation multivariate discordant-pair fractions:
  `4.99e-4` / `1.66e-4`;
- global Spearman rank agreement: greater than `0.9999`;
- confirmation directional accuracy: `0.5168`;
- frozen-pair sign retention: `0.5291`, with block-bootstrap 95% interval
  `[0.4840, 0.5453]`;
- discovery reversals retaining both GC and CS signs: `1 of 5,192`.

Therefore the financial matrix demonstrates that multivariate discordance can
occur, but does not establish a stable market-ranking advantage.

## Relevant files in this release

- `configs/ma_yu_14_index_gaussian_protocol_v1.json`
- `configs/spectralcste_all_edge_mechanism_v1.json`
- `scripts/run_ma_yu_14_index_gaussian.py`
- `scripts/run_spectralcste_all_edge_mechanism.py`
- `data/stock_indexes_canonical.csv`
- `results/ma_yu_14_index_gaussian_protocol_v1/`
- `results/spectralcste_all_edge_mechanism_v1/`

