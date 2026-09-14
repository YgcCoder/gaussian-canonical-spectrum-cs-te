#!/usr/bin/env python3
"""Frozen real-data audit of scalar and multivariate Gaussian CS-TE rankings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from run_gaussian_cs_te_equivalence import (
    gaussian_statistics,
    ma_yu_kernel_score_from_distances,
    official_notebook_median,
    pairwise_squared,
    samplewise_median,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_data(path: Path, expected_hash: str, expected_rows: int, expected_series: int) -> tuple[list[str], np.ndarray]:
    raw = path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"data hash mismatch: {actual_hash}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)[1:]
        rows = [[float(value) for value in row[1:]] for row in reader]
    values = np.asarray(rows, dtype=float)
    if values.shape != (expected_rows, expected_series):
        raise ValueError(f"unexpected data shape: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("canonical data contain non-finite values")
    return header, values


def edge_statistics(values: np.ndarray, source: tuple[int, ...], target: tuple[int, ...]) -> dict[str, float]:
    a = values[:-1, source]
    b = values[1:, target]
    w = values[:-1, target]
    stats = gaussian_statistics(a, b, w)
    return {
        "gc": float(stats["gc"]),
        "gaussian_te": float(stats["te"]),
        "gaussian_cmi": float(stats["te"]),
        "canonical_cs": float(stats["canonical_cs"]),
        "gc_canonical_error": float(stats["gc_vs_canonical_error"]),
    }


def edge_id(names: list[str], source: tuple[int, ...], target: tuple[int, ...]) -> str:
    return "+".join(names[i] for i in source) + "->" + "+".join(names[i] for i in target)


def scalar_edges(names: list[str], values: np.ndarray, split: str) -> list[dict[str, object]]:
    rows = []
    for source in range(len(names)):
        for target in range(len(names)):
            if source == target:
                continue
            stats = edge_statistics(values, (source,), (target,))
            mapped = stats["gc"] / 2.0 + math.log((3.0 + math.exp(-stats["gc"])) / 4.0)
            rows.append({
                "split": split,
                "edge_id": edge_id(names, (source,), (target,)),
                "source": names[source],
                "target": names[target],
                **stats,
                "scalar_mapped_cs": mapped,
                "abs_scalar_map_error": abs(mapped - stats["canonical_cs"]),
            })
    return rows


def block_edges(names: list[str], values: np.ndarray, split: str) -> list[dict[str, object]]:
    rows = []
    pairs = list(itertools.combinations(range(len(names)), 2))
    for source in pairs:
        remaining = [index for index in range(len(names)) if index not in source]
        for target in itertools.combinations(remaining, 2):
            stats = edge_statistics(values, source, target)
            rows.append({
                "split": split,
                "edge_id": edge_id(names, source, target),
                "source_indices": ";".join(map(str, source)),
                "target_indices": ";".join(map(str, target)),
                "source": ";".join(names[i] for i in source),
                "target": ";".join(names[i] for i in target),
                **stats,
            })
    return rows


def rank_values(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(len(order), dtype=float)
    return ranks


def inversion_count(values: list[float]) -> int:
    def merge_count(array: list[float]) -> tuple[list[float], int]:
        if len(array) < 2:
            return array, 0
        middle = len(array) // 2
        left, left_count = merge_count(array[:middle])
        right, right_count = merge_count(array[middle:])
        merged: list[float] = []
        i = j = 0
        count = left_count + right_count
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i]); i += 1
            else:
                merged.append(right[j]); j += 1
                count += len(left) - i
        merged.extend(left[i:]); merged.extend(right[j:])
        return merged, count
    return merge_count(values)[1]


def rank_summary(rows: list[dict[str, object]]) -> dict[str, float | int]:
    ordered = sorted(rows, key=lambda row: (float(row["gc"]), str(row["edge_id"])))
    cs = [float(row["canonical_cs"]) for row in ordered]
    inversions = inversion_count(cs)
    pairs = len(rows) * (len(rows) - 1) // 2
    gc_rank = rank_values([float(row["gc"]) for row in rows])
    cs_rank = rank_values([float(row["canonical_cs"]) for row in rows])
    spearman = float(np.corrcoef(gc_rank, cs_rank)[0, 1])
    return {
        "edges": len(rows),
        "comparable_pairs": pairs,
        "discordant_pairs": inversions,
        "discordant_fraction": inversions / pairs,
        "kendall_tau_no_tie_approx": 1.0 - 2.0 * inversions / pairs,
        "spearman": spearman,
    }


def strongest_reversal(rows: list[dict[str, object]]) -> dict[str, object] | None:
    ordered = sorted(rows, key=lambda row: (float(row["gc"]), str(row["edge_id"])))
    best = None
    max_cs_row = ordered[0]
    for row in ordered[1:]:
        gc_delta = float(row["gc"]) - float(max_cs_row["gc"])
        cs_delta = float(max_cs_row["canonical_cs"]) - float(row["canonical_cs"])
        if gc_delta > 0 and cs_delta > 0:
            candidate = (cs_delta, gc_delta, str(max_cs_row["edge_id"]), str(row["edge_id"]))
            if best is None or candidate > best[0]:
                best = (candidate, max_cs_row, row)
        if float(row["canonical_cs"]) > float(max_cs_row["canonical_cs"]):
            max_cs_row = row
    if best is None:
        return None
    _, low_gc, high_gc = best
    return {
        "low_gc_edge": low_gc["edge_id"],
        "high_gc_edge": high_gc["edge_id"],
        "gc_low": low_gc["gc"],
        "gc_high": high_gc["gc"],
        "cs_low_gc": low_gc["canonical_cs"],
        "cs_high_gc": high_gc["canonical_cs"],
        "gc_gap": float(high_gc["gc"]) - float(low_gc["gc"]),
        "cs_reversal_gap": float(low_gc["canonical_cs"]) - float(high_gc["canonical_cs"]),
    }


def strongest_near_equal(rows: list[dict[str, object]], tolerance: float) -> dict[str, object] | None:
    ordered = sorted(rows, key=lambda row: (float(row["gc"]), str(row["edge_id"])))
    best = None
    for left in range(len(ordered)):
        right = left + 1
        while right < len(ordered) and float(ordered[right]["gc"]) - float(ordered[left]["gc"]) <= tolerance:
            cs_gap = abs(float(ordered[right]["canonical_cs"]) - float(ordered[left]["canonical_cs"]))
            candidate = (cs_gap, str(ordered[left]["edge_id"]), str(ordered[right]["edge_id"]))
            if best is None or candidate > best[0]:
                best = (candidate, ordered[left], ordered[right])
            right += 1
    if best is None:
        return None
    _, one, two = best
    return {
        "edge_one": one["edge_id"],
        "edge_two": two["edge_id"],
        "gc_one": one["gc"],
        "gc_two": two["gc"],
        "absolute_gc_gap": abs(float(one["gc"]) - float(two["gc"])),
        "cs_one": one["canonical_cs"],
        "cs_two": two["canonical_cs"],
        "absolute_cs_gap": abs(float(one["canonical_cs"]) - float(two["canonical_cs"])),
        "tolerance": tolerance,
    }


def row_by_id(rows: list[dict[str, object]], identifier: str) -> dict[str, object]:
    return next(row for row in rows if row["edge_id"] == identifier)


def parse_edge(row: dict[str, object]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    source = tuple(int(value) for value in str(row["source_indices"]).split(";"))
    target = tuple(int(value) for value in str(row["target_indices"]).split(";"))
    return source, target


def bootstrap_confirmation(
    values: np.ndarray,
    low_edge: dict[str, object],
    high_edge: dict[str, object],
    repetitions: int,
    block_length: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    n = len(values)
    low_source, low_target = parse_edge(low_edge)
    high_source, high_target = parse_edge(high_edge)
    rows = []
    for repetition in range(repetitions):
        starts = rng.integers(0, n, size=math.ceil(n / block_length))
        indices = np.concatenate([(start + np.arange(block_length)) % n for start in starts])[:n]
        sample = values[indices]
        low = edge_statistics(sample, low_source, low_target)
        high = edge_statistics(sample, high_source, high_target)
        rows.append({
            "repetition": repetition,
            "gc_low_edge": low["gc"],
            "gc_high_edge": high["gc"],
            "cs_low_edge": low["canonical_cs"],
            "cs_high_edge": high["canonical_cs"],
            "gc_order_retained": int(low["gc"] < high["gc"]),
            "cs_reverse_retained": int(low["canonical_cs"] > high["canonical_cs"]),
            "both_retained": int(low["gc"] < high["gc"] and low["canonical_cs"] > high["canonical_cs"]),
        })
    return rows


def kde_for_edge(values: np.ndarray, row: dict[str, object], eta: float) -> list[dict[str, object]]:
    source, target = parse_edge(row)
    a, b, w = values[:-1, source], values[1:, target], values[:-1, target]
    combined = np.column_stack([w, a, b])
    medians = {
        "official-featurewise": official_notebook_median(combined),
        "samplewise": samplewise_median(combined),
    }
    distances = pairwise_squared(a), pairwise_squared(b), pairwise_squared(w)
    result = []
    for orientation, median in medians.items():
        result.append({
            "edge_id": row["edge_id"],
            "orientation": orientation,
            "eta": eta,
            "median": median,
            "bandwidth": eta * median,
            "gc": row["gc"],
            "gaussian_te": row["gaussian_te"],
            "canonical_cs": row["canonical_cs"],
            "kde_cs": ma_yu_kernel_score_from_distances(*distances, eta * median),
        })
    return result


def main() -> None:
    parsed = parse_args()
    if parsed.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {parsed.output}")
    config_bytes = parsed.config.read_bytes()
    config = json.loads(config_bytes)
    data_path = Path(config["data_path"])
    names, values = load_data(
        data_path,
        str(config["expected_sha256"]),
        int(config["expected_rows"]),
        int(config["expected_series"]),
    )
    parsed.output.mkdir(parents=True, exist_ok=False)

    discovery = values[slice(*config["discovery_rows"])]
    confirmation = values[slice(*config["confirmation_rows"])]
    datasets = {"full": values, "discovery": discovery, "confirmation": confirmation}

    scalar: list[dict[str, object]] = []
    blocks: dict[str, list[dict[str, object]]] = {}
    for split, matrix in datasets.items():
        scalar.extend(scalar_edges(names, matrix, split))
        blocks[split] = block_edges(names, matrix, split)
    write_csv(parsed.output / "scalar_edges.csv", scalar)
    for split, rows in blocks.items():
        write_csv(parsed.output / f"block_edges_{split}.csv", rows)

    positive_gc = [float(row["gc"]) for row in blocks["discovery"] if float(row["gc"]) > 0]
    tolerance = float(config["near_equal_gc_tolerance_fraction_of_median_positive_gc"]) * float(np.median(positive_gc))
    reversal = strongest_reversal(blocks["discovery"])
    near_equal = strongest_near_equal(blocks["discovery"], tolerance)
    if reversal is None:
        bootstrap_rows = []
        kde_rows = []
        confirmation_reversal = None
    else:
        low_discovery = row_by_id(blocks["discovery"], str(reversal["low_gc_edge"]))
        high_discovery = row_by_id(blocks["discovery"], str(reversal["high_gc_edge"]))
        low_confirmation = row_by_id(blocks["confirmation"], str(reversal["low_gc_edge"]))
        high_confirmation = row_by_id(blocks["confirmation"], str(reversal["high_gc_edge"]))
        confirmation_reversal = {
            "low_gc_edge": low_confirmation["edge_id"],
            "high_gc_edge": high_confirmation["edge_id"],
            "gc_low": low_confirmation["gc"],
            "gc_high": high_confirmation["gc"],
            "cs_low_gc": low_confirmation["canonical_cs"],
            "cs_high_gc": high_confirmation["canonical_cs"],
            "both_signs_retained": bool(
                float(low_confirmation["gc"]) < float(high_confirmation["gc"])
                and float(low_confirmation["canonical_cs"]) > float(high_confirmation["canonical_cs"])
            ),
        }
        bootstrap_rows = bootstrap_confirmation(
            confirmation,
            low_discovery,
            high_discovery,
            int(config["bootstrap_repetitions"]),
            int(config["bootstrap_block_length"]),
            int(config["bootstrap_seed"]),
        )
        low_full = row_by_id(blocks["full"], str(reversal["low_gc_edge"]))
        high_full = row_by_id(blocks["full"], str(reversal["high_gc_edge"]))
        kde_rows = kde_for_edge(values, low_full, float(config["kde_eta"]))
        kde_rows += kde_for_edge(values, high_full, float(config["kde_eta"]))
        write_csv(parsed.output / "bootstrap_selected_reversal.csv", bootstrap_rows)
        write_csv(parsed.output / "selected_edge_kde.csv", kde_rows)

    scalar_map_error = max(float(row["abs_scalar_map_error"]) for row in scalar)
    scalar_rank = {}
    for split in datasets:
        subset = [row for row in scalar if row["split"] == split]
        scalar_rank[split] = rank_summary(subset)

    results = {
        "status": "complete",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "data_shape": list(values.shape),
        "series": names,
        "scalar_edge_count_per_split": 182,
        "block_edge_count_per_split": len(blocks["full"]),
        "max_abs_scalar_map_error": scalar_map_error,
        "scalar_rank_summary": scalar_rank,
        "block_rank_summary": {split: rank_summary(rows) for split, rows in blocks.items()},
        "discovery_strongest_reversal": reversal,
        "confirmation_of_selected_reversal": confirmation_reversal,
        "discovery_near_equal_gc": near_equal,
        "bootstrap_both_signs_fraction": (
            float(np.mean([row["both_retained"] for row in bootstrap_rows])) if bootstrap_rows else None
        ),
        "selected_edge_kde": kde_rows,
        "interpretation_boundary": (
            "Exploratory descriptive application on the frozen Ma-Yu matrix; not causal identification."
        ),
    }
    (parsed.output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
