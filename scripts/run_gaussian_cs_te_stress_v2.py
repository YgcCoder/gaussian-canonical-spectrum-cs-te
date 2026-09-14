#!/usr/bin/env python3
"""Random-SPD, stable-VAR, and KDE stress tests for Gaussian CS-TE theory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from run_gaussian_cs_te_equivalence import (
    inv_sqrt,
    ma_yu_kernel_score_from_distances,
    official_notebook_median,
    pairwise_squared,
    samplewise_median,
    simulate_nilpotent_var,
    gaussian_statistics,
)


def solve_discrete_lyapunov_numpy(f: np.ndarray, innovation: np.ndarray) -> np.ndarray:
    """Solve P = F P F' + Q with NumPy under row-major vectorization."""
    n = len(f)
    solution = np.linalg.solve(np.eye(n * n) - np.kron(f, f), innovation.reshape(-1))
    matrix = solution.reshape(n, n)
    return 0.5 * (matrix + matrix.T)


def rank_values(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(len(order), dtype=float)
    return ranks


def spearman(values_a: list[float], values_b: list[float]) -> float:
    return float(np.corrcoef(rank_values(values_a), rank_values(values_b))[0, 1])


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def orthogonal(rng: np.random.Generator, dimension: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    return q


def transform(rng: np.random.Generator, dimension: int, log10_condition: float) -> np.ndarray:
    left = orthogonal(rng, dimension)
    right = orthogonal(rng, dimension)
    exponents = np.linspace(-log10_condition / 2, log10_condition / 2, dimension)
    rng.shuffle(exponents)
    return left @ np.diag(10.0 ** exponents) @ right.T


def covariance_scores(sa: np.ndarray, sb: np.ndarray, cross: np.ndarray) -> dict[str, float | list[float]]:
    p, q = len(sa), len(sb)
    s = np.block([[sa, cross], [cross.T, sb]])
    d = np.block([[sa, np.zeros((p, q))], [np.zeros((q, p)), sb]])
    signs = []
    logs = []
    for matrix in (s + d, s, d):
        sign, value = np.linalg.slogdet(matrix)
        signs.append(sign)
        logs.append(value)
    if min(signs) <= 0:
        raise ValueError("non-positive determinant in Gaussian overlap")
    direct_cs = logs[0] - 0.5 * logs[1] - 0.5 * logs[2] - (p + q) * math.log(2.0)

    full_b = sb - cross.T @ np.linalg.solve(sa, cross)
    sign_b, log_b = np.linalg.slogdet(sb)
    sign_f, log_f = np.linalg.slogdet(full_b)
    if min(sign_b, sign_f) <= 0:
        raise ValueError("non-positive target residual determinant")
    gc = float(log_b - log_f)
    cmi = 0.5 * gc

    chol_a = np.linalg.cholesky(sa)
    chol_b = np.linalg.cholesky(sb)
    r = np.linalg.solve(chol_b, np.linalg.solve(chol_a, cross).T).T
    rho = np.linalg.svd(r, compute_uv=False)
    rho2 = np.clip(rho * rho, 0.0, 1.0 - 1e-14)
    spectral_gc = float(-np.log1p(-rho2).sum())
    spectral_cs = float(np.log1p(-rho2 / 4.0).sum() - 0.5 * np.log1p(-rho2).sum())
    return {
        "gc": gc,
        "gaussian_te": cmi,
        "gaussian_cmi": cmi,
        "direct_cs": float(direct_cs),
        "spectral_gc": spectral_gc,
        "spectral_cs": spectral_cs,
        "max_rho2": float(rho2.max(initial=0.0)),
        "rank": int(np.count_nonzero(rho2 > 1e-10)),
    }


def random_spd_rows(rng: np.random.Generator, config: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    lo, hi = int(config["min_dimension"]), int(config["max_dimension"])
    max_log_cond = float(config["max_transform_log10_condition"])
    for construction in range(int(config["random_spd_constructions"])):
        p, q = int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1))
        k = min(p, q)
        if construction % 12 == 0:
            epsilon = 10.0 ** rng.uniform(-10.0, -4.0)
            rho = np.sort(rng.uniform(0.0, 0.8, size=k))[::-1]
            rho[0] = math.sqrt(1.0 - epsilon)
        else:
            rho = 0.995 * rng.beta(0.8, 1.4, size=k)
        u = orthogonal(rng, p)[:, :k]
        v = orthogonal(rng, q)[:, :k]
        r = u @ np.diag(rho) @ v.T
        log_cond_a = rng.uniform(0.0, max_log_cond)
        log_cond_b = rng.uniform(0.0, max_log_cond)
        pa = transform(rng, p, log_cond_a)
        pb = transform(rng, q, log_cond_b)
        sa, sb, cross = pa @ pa.T, pb @ pb.T, pa @ r @ pb.T
        true_rho2 = rho * rho
        true_gc = float(-np.log1p(-true_rho2).sum())
        true_cs = float(np.log1p(-true_rho2 / 4.0).sum() - 0.5 * np.log1p(-true_rho2).sum())
        base = {
            "construction": construction,
            "p": p,
            "q": q,
            "condition_sa": float(np.linalg.cond(sa)),
            "condition_sb": float(np.linalg.cond(sb)),
            "truth_gc": true_gc,
            "truth_cs": true_cs,
        }
        try:
            score = covariance_scores(sa, sb, cross)
            rows.append({
                **base,
                "status": "ok",
                "error": "",
                **score,
                "abs_gc_spectral_error": abs(float(score["gc"]) - float(score["spectral_gc"])),
                "abs_cs_spectral_error": abs(float(score["direct_cs"]) - float(score["spectral_cs"])),
                "abs_te_cmi_error": abs(float(score["gaussian_te"]) - float(score["gaussian_cmi"])),
                "abs_gc_truth_error": abs(float(score["gc"]) - true_gc),
                "abs_cs_truth_error": abs(float(score["direct_cs"]) - true_cs),
                "abs_spectral_gc_truth_error": abs(float(score["spectral_gc"]) - true_gc),
                "abs_spectral_cs_truth_error": abs(float(score["spectral_cs"]) - true_cs),
            })
        except (ValueError, np.linalg.LinAlgError) as error:
            rows.append({
                **base,
                "status": "numerical_failure",
                "error": str(error),
                "gc": math.nan,
                "gaussian_te": math.nan,
                "gaussian_cmi": math.nan,
                "direct_cs": math.nan,
                "spectral_gc": math.nan,
                "spectral_cs": math.nan,
                "max_rho2": math.nan,
                "rank": 0,
                "abs_gc_spectral_error": math.nan,
                "abs_cs_spectral_error": math.nan,
                "abs_te_cmi_error": math.nan,
                "abs_gc_truth_error": math.nan,
                "abs_cs_truth_error": math.nan,
                "abs_spectral_gc_truth_error": math.nan,
                "abs_spectral_cs_truth_error": math.nan,
            })
    return rows


def random_var_rows(rng: np.random.Generator, config: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    lo, hi = int(config["min_dimension"]), int(config["max_dimension"])
    cap = float(config["var_max_spectral_radius"])
    for construction in range(int(config["random_var_constructions"])):
        p, q = int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1))
        n = p + q
        f = rng.standard_normal((n, n))
        raw_radius = max(abs(np.linalg.eigvals(f)))
        radius = rng.uniform(0.05, cap)
        f *= radius / raw_radius
        innovation_root = rng.standard_normal((n, n)) / math.sqrt(n)
        innovation = innovation_root @ innovation_root.T + 0.2 * np.eye(n)
        state_cov = solve_discrete_lyapunov_numpy(f, innovation)
        ix, iy = np.arange(p), np.arange(p, n)
        fy = f[iy, :]
        pxx = state_cov[np.ix_(ix, ix)]
        pyy = state_cov[np.ix_(iy, iy)]
        pxy = state_cov[np.ix_(ix, iy)]
        cov_xb = state_cov[ix, :] @ fy.T
        cov_bw = fy @ state_cov[:, iy]
        cov_ab = np.block([[pxx, cov_xb], [cov_xb.T, pyy]])
        cov_ab_w = np.vstack([pxy, cov_bw])
        conditional = cov_ab - cov_ab_w @ np.linalg.solve(pyy, cov_ab_w.T)
        sa = conditional[:p, :p]
        sb = conditional[p:, p:]
        cross = conditional[:p, p:]
        base = {
            "construction": construction,
            "p": p,
            "q": q,
            "spectral_radius": float(max(abs(np.linalg.eigvals(f)))),
            "condition_state_cov": float(np.linalg.cond(state_cov)),
        }
        try:
            score = covariance_scores(sa, sb, cross)
            rows.append({
                **base,
                "status": "ok",
                "error": "",
                **score,
                "abs_gc_spectral_error": abs(float(score["gc"]) - float(score["spectral_gc"])),
                "abs_cs_spectral_error": abs(float(score["direct_cs"]) - float(score["spectral_cs"])),
                "abs_te_cmi_error": abs(float(score["gaussian_te"]) - float(score["gaussian_cmi"])),
            })
        except (ValueError, np.linalg.LinAlgError) as error:
            rows.append({
                **base,
                "status": "numerical_failure",
                "error": str(error),
                "gc": math.nan,
                "gaussian_te": math.nan,
                "gaussian_cmi": math.nan,
                "direct_cs": math.nan,
                "spectral_gc": math.nan,
                "spectral_cs": math.nan,
                "max_rho2": math.nan,
                "rank": 0,
                "abs_gc_spectral_error": math.nan,
                "abs_cs_spectral_error": math.nan,
                "abs_te_cmi_error": math.nan,
            })
    return rows


def kde_rows(rng: np.random.Generator, config: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for construction in range(int(config["kde_scalar_constructions"])):
        component_te = float(rng.uniform(0.005, 1.5))
        seed = int(rng.integers(0, 2**31 - 1))
        a, b, w = simulate_nilpotent_var(
            [component_te], int(config["kde_sample_size"]), int(config["kde_burn_in"]), seed
        )
        plug = gaussian_statistics(a, b, w)
        combined = np.column_stack([w, a, b])
        medians = {
            "official-featurewise": official_notebook_median(combined),
            "samplewise": samplewise_median(combined),
        }
        distances = pairwise_squared(a), pairwise_squared(b), pairwise_squared(w)
        for orientation, median in medians.items():
            for eta in config["kde_eta"]:
                kde = ma_yu_kernel_score_from_distances(*distances, float(eta) * median)
                rows.append({
                    "construction": construction,
                    "designed_te": component_te,
                    "seed": seed,
                    "orientation": orientation,
                    "eta": eta,
                    "bandwidth": float(eta) * median,
                    "sample_gc": plug["gc"],
                    "sample_gaussian_te": plug["te"],
                    "sample_canonical_cs": plug["canonical_cs"],
                    "kde_cs": kde,
                })
    return rows


def max_field(rows: list[dict[str, object]], field: str) -> float:
    values = [float(row[field]) for row in rows if np.isfinite(float(row[field]))]
    return max(values)


def max_field_where(rows: list[dict[str, object]], field: str, predicate) -> float:
    values = [
        float(row[field]) for row in rows
        if predicate(row) and np.isfinite(float(row[field]))
    ]
    return max(values)


def main() -> None:
    parsed = args()
    if parsed.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {parsed.output}")
    config_bytes = parsed.config.read_bytes()
    config = json.loads(config_bytes)
    rng = np.random.default_rng(int(config["seed"]))
    parsed.output.mkdir(parents=True, exist_ok=False)

    spd = random_spd_rows(rng, config)
    var = random_var_rows(rng, config)
    kde = kde_rows(rng, config)
    write_csv(parsed.output / "random_spd.csv", spd)
    write_csv(parsed.output / "random_var.csv", var)
    write_csv(parsed.output / "kde_comparison.csv", kde)

    kde_rank = {}
    for orientation in ("official-featurewise", "samplewise"):
        for eta in config["kde_eta"]:
            subset = [r for r in kde if r["orientation"] == orientation and float(r["eta"]) == float(eta)]
            rho = spearman(
                [float(r["sample_gc"]) for r in subset],
                [float(r["kde_cs"]) for r in subset],
            )
            kde_rank[f"{orientation}:eta={eta}"] = float(rho)

    results = {
        "status": "complete",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "random_spd_constructions": len(spd),
        "random_var_constructions": len(var),
        "numerical_failures": {
            "spd": sum(row["status"] != "ok" for row in spd),
            "var": sum(row["status"] != "ok" for row in var),
        },
        "kde_rows": len(kde),
        "max_abs_errors": {
            "spd_gc_vs_spectrum": max_field(spd, "abs_gc_spectral_error"),
            "spd_cs_overlap_vs_spectrum": max_field(spd, "abs_cs_spectral_error"),
            "spd_gc_vs_generation_truth": max_field(spd, "abs_gc_truth_error"),
            "spd_cs_vs_generation_truth": max_field(spd, "abs_cs_truth_error"),
            "spd_spectral_gc_vs_generation_truth": max_field(spd, "abs_spectral_gc_truth_error"),
            "spd_spectral_cs_vs_generation_truth": max_field(spd, "abs_spectral_cs_truth_error"),
            "var_gc_vs_spectrum": max_field(var, "abs_gc_spectral_error"),
            "var_cs_overlap_vs_spectrum": max_field(var, "abs_cs_spectral_error"),
            "te_vs_cmi": max(max_field(spd, "abs_te_cmi_error"), max_field(var, "abs_te_cmi_error")),
        },
        "moderate_condition_max_abs_errors": {
            "definition": "condition_sa and condition_sb <= 1e4; max_rho2 <= 0.99",
            "gc_vs_truth": max_field_where(
                spd, "abs_gc_truth_error",
                lambda row: float(row["condition_sa"]) <= 1e4
                and float(row["condition_sb"]) <= 1e4
                and float(row["max_rho2"]) <= 0.99,
            ),
            "cs_vs_truth": max_field_where(
                spd, "abs_cs_truth_error",
                lambda row: float(row["condition_sa"]) <= 1e4
                and float(row["condition_sb"]) <= 1e4
                and float(row["max_rho2"]) <= 0.99,
            ),
        },
        "worst_conditions": {
            "sa": max_field(spd, "condition_sa"),
            "sb": max_field(spd, "condition_sb"),
            "state_cov": max_field(var, "condition_state_cov"),
            "max_rho2": max(max_field(spd, "max_rho2"), max_field(var, "max_rho2")),
        },
        "kde_spearman_vs_gc": kde_rank,
        "claim_boundary": "KDE correlations are finite-sample diagnostics, not consistency results.",
    }
    (parsed.output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
