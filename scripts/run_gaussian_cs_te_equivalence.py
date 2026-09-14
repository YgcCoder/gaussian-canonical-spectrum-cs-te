#!/usr/bin/env python3
"""Validate population Gaussian CS-TE/GC identities on synthetic VAR data.

This script intentionally reproduces the public Ma--Yu notebook statistic,
including its published bandwidth helper, and also evaluates a samplewise
orientation of the same median heuristic. It refuses to overwrite a run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def inv_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    if values.min() <= 1e-12:
        raise ValueError("Covariance is singular or numerically indefinite")
    return (vectors * (values ** -0.5)) @ vectors.T


def residualize(response: np.ndarray, predictors: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(predictors)), predictors])
    coef, *_ = np.linalg.lstsq(design, response, rcond=None)
    return response - np.einsum("ij,jk->ik", design, coef)


def mle_cov(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    return np.einsum("ni,nj->ij", centered, centered) / len(values)


def gaussian_statistics(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> dict[str, float | list[float]]:
    b_restricted = residualize(b, w)
    b_full = residualize(b, np.column_stack([w, a]))
    cov_restricted = np.atleast_2d(mle_cov(b_restricted))
    cov_full = np.atleast_2d(mle_cov(b_full))
    sign_r, logdet_r = np.linalg.slogdet(cov_restricted)
    sign_f, logdet_f = np.linalg.slogdet(cov_full)
    if sign_r <= 0 or sign_f <= 0:
        raise ValueError("Residual covariance is not positive definite")
    gc = float(logdet_r - logdet_f)
    te = 0.5 * gc

    a_res = residualize(a, w)
    b_res = residualize(b, w)
    sa = np.atleast_2d(mle_cov(a_res))
    sb = np.atleast_2d(mle_cov(b_res))
    a_centered = a_res - a_res.mean(0)
    b_centered = b_res - b_res.mean(0)
    cross = np.einsum("ni,nj->ij", a_centered, b_centered) / len(a_res)
    canonical = inv_sqrt(sa) @ cross @ inv_sqrt(sb)
    rho = np.linalg.svd(canonical, compute_uv=False)
    rho2 = np.clip(rho * rho, 0.0, 1.0 - 1e-12)
    component_te = -0.5 * np.log1p(-rho2)
    cs = np.log1p(-rho2 / 4.0).sum() - 0.5 * np.log1p(-rho2).sum()
    return {
        "gc": gc,
        "te": te,
        "canonical_cs": float(cs),
        "rho2": [float(x) for x in rho2],
        "component_te": [float(x) for x in component_te],
        "gc_vs_canonical_error": float(gc - 2.0 * component_te.sum()),
    }


def pairwise_squared(values: np.ndarray) -> np.ndarray:
    norms = np.sum(values * values, axis=1)
    gram = np.einsum("id,jd->ij", values, values)
    return np.maximum(norms[:, None] + norms[None, :] - 2.0 * gram, 0.0)


def official_notebook_median(values: np.ndarray) -> float:
    # Exact orientation of comp_med in the public notebook: with observations
    # passed as rows, this computes distances between feature columns.
    _, n_columns = values.shape
    norms = np.sum(values * values, axis=0)
    gram = np.einsum("ni,nj->ij", values, values)
    dist2 = norms[:, None] + norms[None, :] - 2.0 * gram
    dist2 = dist2 - np.tril(dist2)
    positive = dist2.reshape(-1)
    positive = positive[positive > 0]
    if not len(positive):
        return float("nan")
    return float(np.sqrt(0.5 * np.median(positive)))


def samplewise_median(values: np.ndarray) -> float:
    dist2 = pairwise_squared(values)
    positive = dist2[np.triu_indices(len(values), k=1)]
    positive = positive[positive > 0]
    if not len(positive):
        return float("nan")
    return float(np.sqrt(0.5 * np.median(positive)))


def ma_yu_kernel_score_from_distances(
    da: np.ndarray, db: np.ndarray, dw: np.ndarray, sigma: float
) -> float:
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    scale = 2.0 * sigma * sigma
    k = np.exp(-da / scale)
    ell = np.exp(-db / scale)
    m = np.exp(-dw / scale)
    h1 = k * m
    h2 = ell * m
    h3 = k * ell * m
    s_m = m.sum(axis=1)
    s_h1 = h1.sum(axis=1)
    s_h2 = h2.sum(axis=1)
    s_h3 = h3.sum(axis=1)
    self1 = float(np.sum(s_h3 * s_m * s_m))
    self2 = float(np.sum((s_h1 * s_h1) * (s_h2 * s_h2) / s_h3))
    cross = float(np.sum(s_m * s_h1 * s_h2))
    if min(self1, self2, cross) <= 0:
        return float("nan")
    return float(-2.0 * np.log(cross) + np.log(self1) + np.log(self2))


def simulate_nilpotent_var(
    component_te: list[float], sample_size: int, burn_in: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dimension = len(component_te)
    coefficient = np.diag([math.sqrt(math.exp(2.0 * x) - 1.0) for x in component_te])
    total = sample_size + burn_in + 1
    x = rng.standard_normal((total, dimension))
    y = np.zeros((total, dimension))
    for t in range(1, total):
        y[t] = coefficient @ x[t - 1] + rng.standard_normal(dimension)
    start = burn_in + 1
    a = x[start - 1 : total - 1]
    b = y[start:total]
    w = y[start - 1 : total - 1]
    return a, b, w


def theoretical_statistics(component_te: list[float]) -> dict[str, float | list[float]]:
    xs = np.asarray(component_te, dtype=float)
    rho2 = 1.0 - np.exp(-2.0 * xs)
    cs_components = xs + np.log((3.0 + np.exp(-2.0 * xs)) / 4.0)
    return {
        "theory_te": float(xs.sum()),
        "theory_gc": float(2.0 * xs.sum()),
        "theory_cs": float(cs_components.sum()),
        "theory_rho2": [float(x) for x in rho2],
    }


def evaluate_dataset(
    model: str,
    component_te: list[float],
    sample_size: int,
    burn_in: int,
    seed: int,
    eta_list: list[float],
    source_scale: float = 1.0,
    experiment_kind: str = "main",
) -> list[dict[str, object]]:
    a, b, w = simulate_nilpotent_var(component_te, sample_size, burn_in, seed)
    a = a * source_scale
    stats = gaussian_statistics(a, b, w)
    combined = np.column_stack([w, a, b])
    med_official = official_notebook_median(combined)
    med_samplewise = samplewise_median(combined)
    da, db, dw = pairwise_squared(a), pairwise_squared(b), pairwise_squared(w)
    theory = theoretical_statistics(component_te)
    rows: list[dict[str, object]] = []
    for eta in eta_list:
        for orientation, median in (("official-featurewise", med_official), ("samplewise", med_samplewise)):
            sigma = eta * median
            row: dict[str, object] = {
                "experiment_kind": experiment_kind,
                "model": model,
                "dimension": len(component_te),
                "component_te_spec": ";".join(str(x) for x in component_te),
                "sample_size": sample_size,
                "seed": seed,
                "source_scale": source_scale,
                "eta": eta,
                "bandwidth_orientation": orientation,
                "median": median,
                "sigma": sigma,
                "kde_cs": ma_yu_kernel_score_from_distances(da, db, dw, sigma),
                **theory,
                "sample_gc": stats["gc"],
                "sample_te": stats["te"],
                "sample_canonical_cs": stats["canonical_cs"],
                "sample_rho2": ";".join(f"{x:.12g}" for x in stats["rho2"]),
                "gc_vs_canonical_error": stats["gc_vs_canonical_error"],
            }
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    keys = (
        "experiment_kind",
        "model",
        "dimension",
        "component_te_spec",
        "sample_size",
        "source_scale",
        "eta",
        "bandwidth_orientation",
    )
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result: list[dict[str, object]] = []
    for group_key, items in groups.items():
        out = dict(zip(keys, group_key))
        out["replicates"] = len(items)
        for field in ("sample_gc", "sample_te", "sample_canonical_cs", "kde_cs"):
            values = np.asarray([float(item[field]) for item in items])
            out[f"{field}_mean"] = float(np.nanmean(values))
            out[f"{field}_sd"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        out["theory_gc"] = items[0]["theory_gc"]
        out["theory_te"] = items[0]["theory_te"]
        out["theory_cs"] = items[0]["theory_cs"]
        out["max_abs_gc_canonical_error"] = max(abs(float(item["gc_vs_canonical_error"])) for item in items)
        result.append(out)
    return sorted(result, key=lambda x: tuple(str(x[k]) for k in keys))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {args.output}")
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    args.output.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, object]] = []
    models: dict[str, list[float]] = {
        f"scalar_te_{value:g}": [float(value)] for value in config["scalar_component_te"]
    }
    models.update({name: [float(x) for x in values] for name, values in config["multivariate_component_te"].items()})
    for model, component_te in models.items():
        for sample_size in config["sample_sizes"]:
            for seed in config["seed_list"]:
                rows.extend(
                    evaluate_dataset(
                        model,
                        component_te,
                        int(sample_size),
                        int(config["burn_in"]),
                        int(seed),
                        [float(x) for x in config["eta_list"]],
                    )
                )

    scale = config["scale_test"]
    for source_scale in scale["source_scales"]:
        rows.extend(
            evaluate_dataset(
                "scalar_scale_test",
                [float(scale["component_te"])],
                int(scale["sample_size"]),
                int(config["burn_in"]),
                int(scale["seed"]),
                [float(scale["eta"])],
                source_scale=float(source_scale),
                experiment_kind="block-rescaling",
            )
        )

    summaries = summarize(rows)
    write_csv(args.output / "replicate_metrics.csv", rows)
    write_csv(args.output / "summary.csv", summaries)

    scalar_errors = []
    for row in rows:
        if int(row["dimension"]) == 1:
            gc = float(row["sample_gc"])
            mapped = 0.5 * gc + math.log((3.0 + math.exp(-gc)) / 4.0)
            scalar_errors.append(abs(mapped - float(row["sample_canonical_cs"])))

    theory = {name: theoretical_statistics(values) for name, values in models.items()}
    result = {
        "experiment": config["experiment"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "status": "completed",
        "row_count": len(rows),
        "summary_row_count": len(summaries),
        "max_abs_gc_vs_partial_canonical_error": max(abs(float(row["gc_vs_canonical_error"])) for row in rows),
        "max_abs_scalar_transform_error": max(scalar_errors),
        "theoretical_models": theory,
        "theoretical_equal_gc_gap": theory["concentrated"]["theory_cs"] - theory["equal_gc_spread"]["theory_cs"],
        "theoretical_order_reversal_gap": theory["concentrated"]["theory_cs"] - theory["larger_gc_spread"]["theory_cs"],
        "notes": [
            "Synthetic nilpotent Gaussian VAR(1); no market data used.",
            "Gaussian TE is computed as one half of fitted log-determinant Geweke GC.",
            "Canonical CS is computed from sample partial canonical correlations.",
            "KDE CS reproduces the public Ma--Yu Gram statistic; bandwidth orientations are reported separately."
        ]
    }
    with (args.output / "results.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
