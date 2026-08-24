"""Run a trained YOLOv8 detector zero-shot over a labeled frame set and turn
per-box detections into per-frame, per-class presence/absence predictions
(the unit the Stage 1 evaluation is scored on: "is instrument X visible in
this frame", not box localization quality).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(images_dir: str | Path) -> list[Path]:
    images_dir = Path(images_dir)
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def predict_presence(
    model_path: str,
    images_dir: str | Path,
    classes: list[str],
    conf_threshold: float = 0.25,
) -> pd.DataFrame:
    """Returns a long-format DataFrame with columns: frame_id, class, predicted (0/1).

    Imports ultralytics lazily so the metrics/comparison code can be used
    (and unit-tested) without the dependency installed.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    name_to_class = {name.lower(): name for name in classes}

    records = []
    for img_path in list_images(images_dir):
        result = model.predict(source=str(img_path), conf=conf_threshold, verbose=False)[0]
        present = set()
        for box in result.boxes:
            label = model.names[int(box.cls)].lower()
            if label in name_to_class:
                present.add(name_to_class[label])
        for cls in classes:
            records.append({"frame_id": img_path.stem, "class": cls, "predicted": int(cls in present)})

    return pd.DataFrame.from_records(records, columns=["frame_id", "class", "predicted"])


def load_expert_labels(labels_csv: str | Path) -> pd.DataFrame:
    """Expected columns: frame_id, class, label (0/1 presence/absence)."""
    df = pd.read_csv(labels_csv)
    required = {"frame_id", "class", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{labels_csv} is missing required columns: {sorted(missing)}")
    df["frame_id"] = df["frame_id"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def align_labels_and_predictions(
    labels_df: pd.DataFrame, predictions_df: pd.DataFrame, classes: list[str]
) -> tuple[dict[str, "pd.Series"], dict[str, "pd.Series"]]:
    """Inner-join predictions onto expert labels per class (frames with no
    expert label are dropped; a warning-worthy mismatch is left to the caller
    to notice via row counts).
    """
    merged = labels_df.merge(predictions_df, on=["frame_id", "class"], how="inner", suffixes=("", "_pred"))
    labels_by_class: dict[str, "pd.Series"] = {}
    predictions_by_class: dict[str, "pd.Series"] = {}
    for cls in classes:
        sub = merged[merged["class"] == cls].sort_values("frame_id")
        labels_by_class[cls] = sub["label"].to_numpy()
        predictions_by_class[cls] = sub["predicted"].to_numpy()
    return labels_by_class, predictions_by_class
