#!/usr/bin/env python3
"""All-edge SpectralCSTE mechanism test under a frozen discovery/confirmation split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_data(path: Path, expected_hash: str, shape: tuple[int, int]) -> tuple[list[str], np.ndarray]:
    raw = path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"data hash mismatch: {actual_hash}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        names = next(reader)[1:]
        values = np.asarray([[float(value) for value in row[1:]] for row in reader])
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"invalid data matrix: shape={values.shape}")
    return names, values


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def phi(value: float | np.ndarray) -> float | np.ndarray:
    return value + np.log((3.0 + np.exp(-2.0 * value)) / 4.0)


def edge_specs(names: list[str], block_size: int) -> list[dict[str, object]]:
    blocks = list(itertools.combinations(range(len(names)), block_size))
    specs = []
    for source in blocks:
        for target in blocks:
            if set(source).intersection(target):
                continue
            identifier = "+".join(names[i] for i in source) + "->" + "+".join(names[i] for i in target)
            specs.append({"edge_id": identifier, "source": source, "target": target})
    return specs


def covariance_matrix(values: np.ndarray) -> np.ndarray:
    lagged = values[:-1]
    future = values[1:]
    joined = np.column_stack([lagged, future])
    centered = joined - joined.mean(axis=0, keepdims=True)
    return np.einsum("ni,nj->ij", centered, centered) / len(centered)


def evaluate_edges(values: np.ndarray, names: list[str], specs: list[dict[str, object]]) -> list[dict[str, object]]:
    covariance = covariance_matrix(values)
    dimension = len(names)
    rows = []
    for spec in specs:
        source = tuple(spec["source"])
        target = tuple(spec["target"])
        b_indices = tuple(dimension + index for index in target)
        u_indices = source + b_indices
        c_uu = covariance[np.ix_(u_indices, u_indices)]
        c_ww = covariance[np.ix_(target, target)]
        c_uw = covariance[np.ix_(u_indices, target)]
        conditional = c_uu - c_uw @ np.linalg.solve(c_ww, c_uw.T)
        p = len(source)
        sa, sb, cross = conditional[:p, :p], conditional[p:, p:], conditional[:p, p:]
        full_b = sb - cross.T @ np.linalg.solve(sa, cross)
        sign_r, log_r = np.linalg.slogdet(sb)
        sign_f, log_f = np.linalg.slogdet(full_b)
        if min(sign_r, sign_f) <= 0:
            raise ValueError(f"non-positive residual determinant: {spec['edge_id']}")
        gc = float(log_r - log_f)
        chol_a, chol_b = np.linalg.cholesky(sa), np.linalg.cholesky(sb)
        whitened = np.linalg.solve(chol_b, np.linalg.solve(chol_a, cross).T).T
        rho2 = np.clip(np.linalg.svd(whitened, compute_uv=False) ** 2, 0.0, 1.0 - 1e-14)
        components = -0.5 * np.log1p(-rho2)
        spectral_gc = float(2.0 * components.sum())
        cs = float(np.asarray(phi(components)).sum())
        total = float(components.sum())
        if len(components) == 1:
            spread = 0.0
            rank_one_gap = 0.0
            normalized_gap = 0.0
        else:
            spread = float(4.0 * components[0] * components[1] / (total * total)) if total > 0 else 0.0
            rank_one_gap = float(phi(total) - cs)
            maximum_gap = float(phi(total) - 2.0 * phi(total / 2.0))
            normalized_gap = rank_one_gap / maximum_gap if maximum_gap > 1e-18 else 0.0
        rows.append({
            "edge_id": spec["edge_id"],
            "source": ";".join(names[i] for i in source),
            "target": ";".join(names[i] for i in target),
            "gc": gc,
            "gaussian_te_cmi": gc / 2.0,
            "canonical_cs": cs,
            "component_te_1": float(components[0]),
            "component_te_2": float(components[1]) if len(components) > 1 else 0.0,
            "spectrum_spread_h": spread,
            "rank_one_gap_q": rank_one_gap,
            "normalized_gap_qn": normalized_gap,
            "abs_gc_spectrum_error": abs(gc - spectral_gc),
        })
    return rows


def ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    result = np.empty(len(order), dtype=float)
    result[order] = np.arange(len(order), dtype=float)
    return result


def association(rows: list[dict[str, object]]) -> dict[str, float]:
    spread = np.asarray([float(row["spectrum_spread_h"]) for row in rows])
    gap = np.asarray([float(row["normalized_gap_qn"]) for row in rows])
    pearson = float(np.corrcoef(spread, gap)[0, 1])
    spearman = float(np.corrcoef(ranks(spread.tolist()), ranks(gap.tolist()))[0, 1])
    return {
        "pearson_h_qn": pearson,
        "spearman_h_qn": spearman,
        "standardized_ols_slope_h_to_qn": pearson,
        "mean_h": float(spread.mean()),
        "mean_q": float(np.mean([float(row["rank_one_gap_q"]) for row in rows])),
        "mean_qn": float(gap.mean()),
    }


def near_pairs(rows: list[dict[str, object]], tolerance: float) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: (float(row["gc"]), str(row["edge_id"])))
    pairs = []
    for left_index, low in enumerate(ordered):
        right_index = left_index + 1
        while right_index < len(ordered):
            high = ordered[right_index]
            gc_delta = float(high["gc"]) - float(low["gc"])
            if gc_delta > tolerance:
                break
            if gc_delta > 0:
                h_delta = float(high["spectrum_spread_h"]) - float(low["spectrum_spread_h"])
                cs_delta = float(low["canonical_cs"]) - float(high["canonical_cs"])
                pairs.append({
                    "low_gc_edge": low["edge_id"],
                    "high_gc_edge": high["edge_id"],
                    "gc_delta": gc_delta,
                    "h_high_minus_low": h_delta,
                    "cs_low_minus_high": cs_delta,
                })
            right_index += 1
    return pairs


def pair_summary(pairs: list[dict[str, object]], kappa: float) -> dict[str, float | int]:
    qualified = [pair for pair in pairs if abs(float(pair["h_high_minus_low"])) >= kappa]
    predicted = [pair for pair in qualified if float(pair["h_high_minus_low"]) >= kappa]
    opposite = [pair for pair in qualified if float(pair["h_high_minus_low"]) <= -kappa]

    def reversal_rate(group: list[dict[str, object]]) -> float:
        return float(np.mean([float(pair["cs_low_minus_high"]) > 0 for pair in group])) if group else math.nan

    predicted_rate = reversal_rate(predicted)
    opposite_rate = reversal_rate(opposite)
    correct = []
    for pair in qualified:
        predicted_cs_sign = 1.0 if float(pair["h_high_minus_low"]) > 0 else -1.0
        actual_cs_sign = np.sign(float(pair["cs_low_minus_high"]))
        correct.append(predicted_cs_sign == actual_cs_sign)
    return {
        "near_pair_count": len(pairs),
        "qualified_pair_count": len(qualified),
        "directional_accuracy": float(np.mean(correct)) if correct else math.nan,
        "predicted_reversal_pair_count": len(predicted),
        "predicted_reversal_fraction": predicted_rate,
        "predicted_mean_signed_cs_gap": (
            float(np.mean([float(pair["cs_low_minus_high"]) for pair in predicted])) if predicted else math.nan
        ),
        "opposite_pair_count": len(opposite),
        "opposite_reversal_fraction": opposite_rate,
        "reversal_risk_difference": predicted_rate - opposite_rate,
    }


def membership_rows(pairs: list[dict[str, object]], kappa: float) -> list[dict[str, object]]:
    result = []
    for pair in pairs:
        h_delta = float(pair["h_high_minus_low"])
        if abs(h_delta) < kappa:
            continue
        result.append({
            **pair,
            "predicted_cs_sign": 1 if h_delta > 0 else -1,
            "discovery_observed_reversal": int(float(pair["cs_low_minus_high"]) > 0),
        })
    return result


def temporal_summary(members: list[dict[str, object]], confirmation: list[dict[str, object]]) -> dict[str, float | int]:
    lookup = {str(row["edge_id"]): row for row in confirmation}
    direction_correct = []
    observed_reversal_retained = []
    for pair in members:
        low = lookup[str(pair["low_gc_edge"])]
        high = lookup[str(pair["high_gc_edge"])]
        cs_sign = np.sign(float(low["canonical_cs"]) - float(high["canonical_cs"]))
        direction_correct.append(cs_sign == int(pair["predicted_cs_sign"]))
        if int(pair["discovery_observed_reversal"]):
            retained = (
                float(low["gc"]) < float(high["gc"])
                and float(low["canonical_cs"]) > float(high["canonical_cs"])
            )
            observed_reversal_retained.append(retained)
    return {
        "discovery_qualified_pairs": len(members),
        "confirmation_predicted_sign_retention": float(np.mean(direction_correct)),
        "discovery_observed_reversal_pairs": len(observed_reversal_retained),
        "confirmation_both_signs_retention": (
            float(np.mean(observed_reversal_retained)) if observed_reversal_retained else math.nan
        ),
    }


def scalar_sanity(values: np.ndarray, names: list[str]) -> dict[str, float | int]:
    rows = evaluate_edges(values, names, edge_specs(names, 1))
    ordered = sorted(rows, key=lambda row: float(row["gc"]))
    cs = [float(row["canonical_cs"]) for row in ordered]
    inversions = sum(cs[i] > cs[j] for i in range(len(cs)) for j in range(i + 1, len(cs)))
    errors = []
    for row in rows:
        gc = float(row["gc"])
        mapped = gc / 2.0 + math.log((3.0 + math.exp(-gc)) / 4.0)
        errors.append(abs(mapped - float(row["canonical_cs"])))
    return {"edges": len(rows), "discordant_pairs": inversions, "max_abs_map_error": max(errors)}


def percentile_interval(values: list[float], percentiles: list[float]) -> list[float]:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    return [float(value) for value in np.percentile(finite, percentiles)]


def main() -> None:
    parsed = parse_args()
    if parsed.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {parsed.output}")
    config_bytes = parsed.config.read_bytes()
    config = json.loads(config_bytes)
    names, values = load_data(
        Path(config["data_path"]),
        str(config["expected_sha256"]),
        (int(config["expected_rows"]), int(config["expected_series"])),
    )
    parsed.output.mkdir(parents=True, exist_ok=False)
    specs = edge_specs(names, 2)
    discovery_values = values[slice(*config["discovery_rows"])]
    confirmation_values = values[slice(*config["confirmation_rows"])]
    discovery = evaluate_edges(discovery_values, names, specs)
    confirmation = evaluate_edges(confirmation_values, names, specs)
    write_csv(parsed.output / "edges_discovery.csv", discovery)
    write_csv(parsed.output / "edges_confirmation.csv", confirmation)

    median_positive_gc = float(np.median([float(row["gc"]) for row in discovery if float(row["gc"]) > 0]))
    tau = float(config["near_gc_fraction_of_discovery_median_positive_gc"]) * median_positive_gc
    discovery_near = near_pairs(discovery, tau)
    nonzero_h_differences = [
        abs(float(pair["h_high_minus_low"]))
        for pair in discovery_near if abs(float(pair["h_high_minus_low"])) > 0
    ]
    kappa = float(np.quantile(nonzero_h_differences, float(config["dispersion_difference_quantile_within_near_gc"])))
    members = membership_rows(discovery_near, kappa)
    write_csv(parsed.output / "discovery_pair_membership.csv", members)

    confirmation_near = near_pairs(confirmation, tau)
    discovery_summary = pair_summary(discovery_near, kappa)
    confirmation_summary = pair_summary(confirmation_near, kappa)
    temporal = temporal_summary(members, confirmation)

    bootstrap_rows = []
    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    n = len(confirmation_values)
    block_length = int(config["bootstrap_block_length"])
    failures = 0
    for repetition in range(int(config["bootstrap_repetitions"])):
        try:
            starts = rng.integers(0, n, size=math.ceil(n / block_length))
            indices = np.concatenate([(start + np.arange(block_length)) % n for start in starts])[:n]
            sample_edges = evaluate_edges(confirmation_values[indices], names, specs)
            sample_association = association(sample_edges)
            sample_pairs = pair_summary(near_pairs(sample_edges, tau), kappa)
            sample_temporal = temporal_summary(members, sample_edges)
            bootstrap_rows.append({
                "repetition": repetition,
                "status": "ok",
                "standardized_slope": sample_association["standardized_ols_slope_h_to_qn"],
                "directional_accuracy": sample_pairs["directional_accuracy"],
                "predicted_reversal_fraction": sample_pairs["predicted_reversal_fraction"],
                "reversal_risk_difference": sample_pairs["reversal_risk_difference"],
                "temporal_predicted_sign_retention": sample_temporal["confirmation_predicted_sign_retention"],
                "temporal_both_signs_retention": sample_temporal["confirmation_both_signs_retention"],
            })
        except (ValueError, np.linalg.LinAlgError) as error:
            failures += 1
            bootstrap_rows.append({
                "repetition": repetition,
                "status": f"failure:{error}",
                "standardized_slope": math.nan,
                "directional_accuracy": math.nan,
                "predicted_reversal_fraction": math.nan,
                "reversal_risk_difference": math.nan,
                "temporal_predicted_sign_retention": math.nan,
                "temporal_both_signs_retention": math.nan,
            })
        if (repetition + 1) % 20 == 0:
            print(f"bootstrap {repetition + 1}/{config['bootstrap_repetitions']}", flush=True)
    write_csv(parsed.output / "bootstrap_metrics.csv", bootstrap_rows)

    percentiles = [float(value) for value in config["ci_percentiles"]]
    ci = {}
    for field in (
        "standardized_slope",
        "directional_accuracy",
        "predicted_reversal_fraction",
        "reversal_risk_difference",
        "temporal_predicted_sign_retention",
        "temporal_both_signs_retention",
    ):
        ci[field] = percentile_interval([float(row[field]) for row in bootstrap_rows], percentiles)

    results = {
        "status": "complete",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "data_sha256": hashlib.sha256(Path(config["data_path"]).read_bytes()).hexdigest(),
        "edge_count": len(specs),
        "tau_near_gc": tau,
        "kappa_dispersion": kappa,
        "discovery_association": association(discovery),
        "confirmation_association": association(confirmation),
        "discovery_pair_summary": discovery_summary,
        "confirmation_pair_summary": confirmation_summary,
        "temporal_stability": temporal,
        "scalar_sanity": {
            "discovery": scalar_sanity(discovery_values, names),
            "confirmation": scalar_sanity(confirmation_values, names),
        },
        "bootstrap_repetitions": len(bootstrap_rows),
        "bootstrap_failures": failures,
        "bootstrap_percentile_95_ci": ci,
        "decision_rule_note": "Financial support requires positive confirmation directions, non-null bootstrap intervals, and nontrivial temporal stability.",
    }
    (parsed.output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
