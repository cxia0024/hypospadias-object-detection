"""Build the Stage 1 comparison table across an arbitrary set of datasets
(AVOS test split, hypospadias zero-shot eval, and any additional comparison
dataset such as a held-out AVOS procedure or an external instrument-detection
benchmark) and the pairwise statistical tests between them.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

from .metrics import DatasetEvalResult, compare_two_datasets


def per_class_table(results: list[DatasetEvalResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        for cls, m in result.per_class.items():
            lo, hi = m.wilson_ci
            rows.append(
                {
                    "dataset": result.dataset_name,
                    "class": cls,
                    "n": m.counts.n,
                    "accuracy": m.accuracy,
                    "wilson_ci_low": lo,
                    "wilson_ci_high": hi,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "tp": m.counts.tp,
                    "fp": m.counts.fp,
                    "tn": m.counts.tn,
                    "fn": m.counts.fn,
                    "binomial_p_vs_chance": m.binomial_p_vs_chance,
                }
            )
    return pd.DataFrame(rows)


def pooled_table(results: list[DatasetEvalResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        c = result.pooled_counts
        lo, hi = result.pooled_wilson_ci()
        rows.append(
            {
                "dataset": result.dataset_name,
                "n": c.n,
                "accuracy": result.pooled_accuracy,
                "wilson_ci_low": lo,
                "wilson_ci_high": hi,
                "binomial_p_vs_chance": result.pooled_binomial_p_vs_chance(),
            }
        )
    return pd.DataFrame(rows)


def pairwise_comparison_table(results: list[DatasetEvalResult]) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(results, 2):
        cmp = compare_two_datasets(a, b)
        rows.append(
            {
                "dataset_a": cmp["dataset_a"],
                "dataset_b": cmp["dataset_b"],
                "class": "pooled",
                "z": cmp["pooled"]["z"],
                "p_value": cmp["pooled"]["p_value"],
            }
        )
        for cls, stat in cmp["per_class"].items():
            rows.append(
                {
                    "dataset_a": cmp["dataset_a"],
                    "dataset_b": cmp["dataset_b"],
                    "class": cls,
                    "z": stat["z"],
                    "p_value": stat["p_value"],
                }
            )
    return pd.DataFrame(rows)


def write_report(results: list[DatasetEvalResult], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_class_table(results).to_csv(out_dir / "per_class_metrics.csv", index=False)
    pooled_table(results).to_csv(out_dir / "pooled_metrics.csv", index=False)
    pairwise_comparison_table(results).to_csv(out_dir / "pairwise_comparisons.csv", index=False)

    summary = {
        "datasets": [r.dataset_name for r in results],
        "pooled_accuracy": {r.dataset_name: r.pooled_accuracy for r in results},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
