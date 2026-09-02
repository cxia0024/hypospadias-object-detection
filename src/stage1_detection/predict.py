"""Run a trained YOLOv8 detector zero-shot over a labeled frame set and turn
per-box detections into per-frame, per-class presence/absence predictions
(the unit the Stage 1 evaluation is scored on: "is instrument X visible in
this frame", not box localization quality).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(images_dir: str | Path, exclude_prefixes: list[str] | None = None) -> list[Path]:
    """`exclude_prefixes`, if given, drops any image whose filename stem
    *starts with* one of the given prefixes -- e.g. to drop a specific
    source video from both inference and scoring without moving or deleting
    the underlying files. Prefix (not "contains anywhere") matching so
    excluding video 3 doesn't also catch video 30 or video 13 -- include the
    separator in the prefix (e.g. "3_" rather than bare "3") if filenames are
    like "3_frame001.jpg", since "3_" doesn't prefix-match "30_frame001.jpg".
    """
    images_dir = Path(images_dir)
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if exclude_prefixes:
        images = [p for p in images if not any(p.stem.startswith(prefix) for prefix in exclude_prefixes)]
    return images


def get_model_classes(model_path: str) -> list[str]:
    """The AVOS class names the checkpoint was actually trained on, in index order.

    Use this instead of hardcoding a class list -- the model is the source of truth.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    return [model.names[i] for i in sorted(model.names)]


def predict_presence(
    model_path: str,
    images_dir: str | Path,
    classes: list[str] | None = None,
    conf_threshold: float = 0.25,
    exclude_prefixes: list[str] | None = None,
) -> pd.DataFrame:
    """Returns a long-format DataFrame with columns: frame_id, class, predicted (0/1).

    Imports ultralytics lazily so the metrics/comparison code can be used
    (and unit-tested) without the dependency installed.

    `classes` defaults to the checkpoint's own class list (via `get_model_classes`);
    pass an explicit subset only if you want to score fewer classes than the model has.
    `exclude_prefixes` drops matching frames before running inference -- see `list_images`.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    if classes is None:
        classes = [model.names[i] for i in sorted(model.names)]
    name_to_class = {name.lower(): name for name in classes}

    records = []
    for img_path in list_images(images_dir, exclude_prefixes=exclude_prefixes):
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
    """Expected columns: frame_id, class, label (0/1 presence/absence).

    Rejects a file with any blank/non-0-1 label cells -- that's an unfinished
    labeling template (see extract_frames.write_labeling_template), not
    expert ground truth, and must not be silently scored as one.
    """
    df = pd.read_csv(labels_csv)
    required = {"frame_id", "class", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{labels_csv} is missing required columns: {sorted(missing)}")

    unlabeled_mask = df["label"].isna() | (df["label"].astype(str).str.strip() == "")
    n_unlabeled = int(unlabeled_mask.sum())
    if n_unlabeled:
        raise ValueError(
            f"{labels_csv} has {n_unlabeled}/{len(df)} rows with no label -- "
            "this looks like an unfinished labeling template. An expert must fill in "
            "presence/absence (0/1) for every frame/class before this file can be used "
            "as labels_csv in the Stage 1 evaluation."
        )

    df["frame_id"] = df["frame_id"].astype(str)
    try:
        df["label"] = df["label"].astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{labels_csv} has non-0/1 values in the 'label' column: {exc}") from exc
    if not df["label"].isin([0, 1]).all():
        bad = sorted(df.loc[~df["label"].isin([0, 1]), "label"].unique())
        raise ValueError(f"{labels_csv} 'label' column must be 0/1 only, found: {bad}")

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
