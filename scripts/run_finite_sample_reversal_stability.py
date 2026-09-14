#!/usr/bin/env python3
"""Estimate recovery of the three canonical-spectrum counterexamples by sample size."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from run_gaussian_cs_te_equivalence import (
    gaussian_statistics,
    simulate_nilpotent_var,
    theoretical_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - half, center + half


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

    models = {name: [float(x) for x in values] for name, values in config["models"].items()}
    required = {"A_concentrated", "B_equal_gc_spread", "C_larger_gc_spread"}
    if set(models) != required:
        raise ValueError(f"Expected exactly {sorted(required)}")

    theory = {name: theoretical_statistics(values) for name, values in models.items()}
    if not math.isclose(theory["A_concentrated"]["theory_gc"], theory["B_equal_gc_spread"]["theory_gc"]):
        raise ValueError("A and B must have equal population GC")
    if not theory["A_concentrated"]["theory_cs"] > theory["B_equal_gc_spread"]["theory_cs"]:
        raise ValueError("A and B must have unequal population CS-TE in the intended direction")
    if not (
        theory["A_concentrated"]["theory_gc"] < theory["C_larger_gc_spread"]["theory_gc"]
        and theory["A_concentrated"]["theory_cs"] > theory["C_larger_gc_spread"]["theory_cs"]
    ):
        raise ValueError("A and C must form the intended population order reversal")

    rng = np.random.default_rng(int(config["master_seed"]))
    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    total_failures = 0

    for sample_size in [int(x) for x in config["sample_sizes"]]:
        equal_gc_successes = 0
        gc_order_successes = 0
        cs_order_successes = 0
        joint_successes = 0
        completed = 0
        failures = 0

        for replicate in range(int(config["replicates"])):
            model_stats: dict[str, dict[str, float | list[float]]] = {}
            seeds: dict[str, int] = {}
            try:
                for model_name, component_te in models.items():
                    seed = int(rng.integers(0, np.iinfo(np.int32).max))
                    seeds[model_name] = seed
                    a, b, w = simulate_nilpotent_var(
                        component_te,
                        sample_size,
                        int(config["burn_in"]),
                        seed,
                    )
                    model_stats[model_name] = gaussian_statistics(a, b, w)
            except (ValueError, np.linalg.LinAlgError):
                failures += 1
                total_failures += 1
                continue

            a_stats = model_stats["A_concentrated"]
            b_stats = model_stats["B_equal_gc_spread"]
            c_stats = model_stats["C_larger_gc_spread"]
            equal_gc_recovered = float(a_stats["canonical_cs"]) > float(b_stats["canonical_cs"])
            gc_order_recovered = float(a_stats["gc"]) < float(c_stats["gc"])
            cs_order_recovered = float(a_stats["canonical_cs"]) > float(c_stats["canonical_cs"])
            joint_recovered = gc_order_recovered and cs_order_recovered

            equal_gc_successes += int(equal_gc_recovered)
            gc_order_successes += int(gc_order_recovered)
            cs_order_successes += int(cs_order_recovered)
            joint_successes += int(joint_recovered)
            completed += 1
            raw_rows.append(
                {
                    "sample_size": sample_size,
                    "replicate": replicate,
                    "seed_a": seeds["A_concentrated"],
                    "seed_b": seeds["B_equal_gc_spread"],
                    "seed_c": seeds["C_larger_gc_spread"],
                    "gc_a": a_stats["gc"],
                    "cs_a": a_stats["canonical_cs"],
                    "gc_b": b_stats["gc"],
                    "cs_b": b_stats["canonical_cs"],
                    "gc_c": c_stats["gc"],
                    "cs_c": c_stats["canonical_cs"],
                    "equal_gc_cs_separation_recovered": int(equal_gc_recovered),
                    "gc_order_recovered": int(gc_order_recovered),
                    "cs_order_recovered": int(cs_order_recovered),
                    "joint_reversal_recovered": int(joint_recovered),
                }
            )

        if completed == 0:
            raise RuntimeError(f"No completed replicates for sample size {sample_size}")

        summary: dict[str, object] = {
            "sample_size": sample_size,
            "planned_replicates": int(config["replicates"]),
            "completed_replicates": completed,
            "failures": failures,
        }
        for metric_name, successes in (
            ("equal_gc_cs_separation", equal_gc_successes),
            ("gc_order", gc_order_successes),
            ("cs_order", cs_order_successes),
            ("joint_reversal", joint_successes),
        ):
            low, high = wilson_interval(successes, completed)
            summary[f"{metric_name}_successes"] = successes
            summary[f"{metric_name}_rate"] = successes / completed
            summary[f"{metric_name}_ci_low"] = low
            summary[f"{metric_name}_ci_high"] = high
        summary_rows.append(summary)

    write_csv(args.output / "replicate_metrics.csv", raw_rows)
    write_csv(args.output / "recovery_by_sample_size.csv", summary_rows)

    result = {
        "experiment": config["experiment"],
        "status": "completed",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "master_seed": int(config["master_seed"]),
        "sample_sizes": [int(x) for x in config["sample_sizes"]],
        "replicates_per_sample_size": int(config["replicates"]),
        "total_completed": len(raw_rows),
        "total_failures": total_failures,
        "theoretical_models": theory,
        "summary": summary_rows,
        "boundary": "Three fixed nilpotent Gaussian VAR constructions and the Gaussian covariance plug-in only; not a general power theorem and not a KDE evaluation.",
    }
    with (args.output / "results.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
