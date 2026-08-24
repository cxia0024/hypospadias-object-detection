#!/usr/bin/env python3
"""CLI entrypoint for the Stage 1 zero-shot evaluation + cross-dataset comparison.

Usage:
    python scripts/run_stage1_eval.py --config configs/stage1_datasets.yaml --out results/stage1

For each dataset in the config, runs the YOLOv8 detector over its frames,
scores per-class presence/absence against the expert labels, and writes:
    per_class_metrics.csv     -- accuracy/precision/recall/F1/Wilson CI/binomial p, per class per dataset
    pooled_metrics.csv        -- same, pooled across classes, per dataset
    pairwise_comparisons.csv  -- two-proportion z-test between every pair of datasets
    summary.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stage1_detection.metrics import evaluate_dataset  # noqa: E402
from stage1_detection.predict import (  # noqa: E402
    align_labels_and_predictions,
    get_model_classes,
    load_expert_labels,
    predict_presence,
)
from stage1_detection.report import write_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to stage1_datasets.yaml")
    parser.add_argument("--out", default="results/stage1", help="Output directory for the comparison report")
    parser.add_argument(
        "--datasets",
        default=None,
        help=(
            "Comma-separated subset of dataset names from the config to run (e.g. avos_test). "
            "Default: all datasets in the config. Useful when some datasets (e.g. hypospadias_eval) "
            "aren't expert-labeled yet -- pairwise comparisons only cover whatever subset you run."
        ),
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    model_path: str = config["model_path"]
    conf_threshold: float = config.get("conf_threshold", 0.25)

    classes: list[str] = config.get("classes") or get_model_classes(model_path)
    print(f"Classes (from {'config' if config.get('classes') else 'model checkpoint'}): {classes}")

    all_datasets = config["datasets"]
    if args.datasets:
        wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
        unknown = [d for d in wanted if d not in all_datasets]
        if unknown:
            raise SystemExit(f"unknown dataset name(s) in --datasets: {unknown}. Config has: {list(all_datasets)}")
        selected_datasets = {name: all_datasets[name] for name in wanted}
    else:
        selected_datasets = all_datasets

    results = []
    for name, ds_cfg in selected_datasets.items():
        print(f"[{name}] running inference over {ds_cfg['images_dir']} ...")
        predictions_df = predict_presence(model_path, ds_cfg["images_dir"], classes, conf_threshold)
        labels_df = load_expert_labels(ds_cfg["labels_csv"])
        labels_by_class, predictions_by_class = align_labels_and_predictions(labels_df, predictions_df, classes)

        n_frames = {cls: len(labels_by_class[cls]) for cls in classes}
        print(f"[{name}] scored frames per class: {n_frames}")

        chance = ds_cfg.get("chance_level", 0.5)
        results.append(evaluate_dataset(name, labels_by_class, predictions_by_class, chance=chance))

    write_report(results, args.out)
    print(f"Wrote comparison report to {args.out}/")


if __name__ == "__main__":
    main()
