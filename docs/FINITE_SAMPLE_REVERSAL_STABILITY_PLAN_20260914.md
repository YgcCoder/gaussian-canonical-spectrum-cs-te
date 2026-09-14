# Finite-sample reversal-stability experiment plan

Date: 2026-09-14 Asia/Singapore

## Gap-to-RQ alignment

| field | frozen value |
|---|---|
| core problem | The population construction proves that GC and CS-TE rankings can reverse, but the existing numerical audit does not show when the reversal is recoverable from estimated covariances. |
| gap ID | E-FINITE-REVERSAL |
| evidence the gap exists | The current paper reports population identities and machine-precision checks only; Table 1 gives no sampling distribution. |
| required capability | Quantify recovery of the two Table-1 ordering consequences as sample size increases. |
| contribution/mechanism | Strict Schur-convexity of CS-TE across partial canonical modes. |
| primary RQ | At what sample sizes does the Gaussian covariance plug-in recover the population equal-G separation and GC/CS-TE order reversal? |
| scientific unit | One independently simulated pair of stable nilpotent Gaussian VAR(1) systems. |
| data and split | Synthetic only; no train/test split and no financial data. |
| comparator/intervention | Sample size; the two population inequalities are the fixed oracle. |
| information parity | Every model uses the same estimator and sample size; model-specific random streams are independent. |
| metrics | Fraction of replicates recovering `CS_A > CS_B`; fraction recovering both `GC_A < GC_C` and `CS_A > CS_C`; Wilson 95% intervals over all replicates. |
| expected pattern | Recovery should increase with sample size and approach one if the plug-in estimator resolves the fixed population margins. |
| evidence artifact | Raw replicate CSV, sample-size summary CSV, results JSON, and a data-derived two-panel figure. |
| finding boundary | This tests only the three feasible Gaussian VAR constructions and the covariance plug-in; it is not a power theorem for arbitrary spectra or the Ma--Yu KDE statistic. |
| stop/pivot rule | If failures or invalid covariance estimates exceed 1%, do not plot a recovery curve until the estimator path is diagnosed. If recovery is non-monotone beyond Wilson uncertainty, report the behavior rather than smoothing it away. |

## Frozen population constructions

- A, concentrated: `x=(1.00,0.00)`.
- B, equal-G spread: `x=(0.50,0.50)`.
- C, larger-G spread: `x=(0.55,0.55)`.
- Stable realization: `X_t ~ N(0,I)` independently and
  `Y_t = C X_{t-1} + epsilon_t`, with
  `C=diag(sqrt(exp(2x_j)-1))` and `epsilon_t ~ N(0,I)`.

## Frozen run design

- sample sizes: `64, 128, 256, 512, 1024, 2048, 4096`;
- replicates per sample size: `1000`;
- burn-in: `64`;
- master seed: `20260914`;
- no reselection after observing results.
