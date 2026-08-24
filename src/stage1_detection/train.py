"""Fine-tune a COCO-pretrained YOLOv8 checkpoint on the AVOS bounding-box
dataset, starting from an already-prepared train/val split.

This does NOT create or modify your split -- it expects a standard
Ultralytics YOLO dataset config (`data.yaml`) with `train:`, `val:`, and
`names:` keys pointing at it. Class names come entirely from that file;
nothing here hardcodes AVOS's instrument classes.

Usage:
    python -m stage1_detection.train --data path/to/avos_data.yaml --epochs 100

After training, the best checkpoint is copied to --out (default
models/yolov8_avos_best.pt) so it's immediately usable as `model_path` in
configs/stage1_datasets.yaml for the Stage 1 zero-shot evaluation.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_to_label_path(img_path: Path) -> Path | None:
    """Mirrors ultralytics' own label-path derivation: it replaces the LAST
    '<sep>images<sep>' segment in an image's path with '<sep>labels<sep>' and
    swaps the extension for .txt. Returns None if the image path has no such
    segment -- e.g. a folder named 'images_split' instead of 'images' -- in
    which case ultralytics silently finds no label for ANY image and treats
    the whole split as unlabeled "background", rather than raising early.
    """
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    img_str = str(img_path)
    if sa not in img_str:
        return None
    label_str = sb.join(img_str.rsplit(sa, 1))
    return Path(label_str).with_suffix(".txt")


def _check_labels_resolvable(split_name: str, split_path: Path) -> None:
    images = sorted(p for p in split_path.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise FileNotFoundError(f"'{split_name}' path {split_path} contains no image files")

    sample = images[0]
    label_path = image_to_label_path(sample)
    if label_path is None:
        raise ValueError(
            f"'{split_name}' images live under {split_path}, which has no '{os.sep}images{os.sep}' "
            "path segment. Ultralytics finds labels by replacing '/images/' with '/labels/' in each "
            "image's path -- with a folder named e.g. 'images_split' instead of 'images', that "
            "substitution silently fails and every image gets scored as an unlabeled background. "
            "Rename the images folder (and its label counterpart) so 'images'/'labels' appear as "
            "literal path components, e.g. images_split/train -> images/train and "
            "labels_split/train -> labels/train, then update data.yaml's train/val paths to match."
        )

    n_with_labels = sum(1 for img in images if image_to_label_path(img).exists())
    if n_with_labels == 0:
        raise FileNotFoundError(
            f"'{split_name}': found {len(images)} images under {split_path}, but no matching label "
            f".txt files at the expected location (e.g. {label_path}). Check that your labels folder "
            "mirrors the images folder's structure (same subfolder names, same base filenames)."
        )


def validate_data_yaml(data_path: str | Path) -> dict:
    """Sanity-checks a YOLO dataset config before handing it to ultralytics --
    a typo here otherwise surfaces as a confusing failure hours into training.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    config = yaml.safe_load(data_path.read_text())
    required = {"train", "val", "names"}
    missing = required - set(config or {})
    if missing:
        raise ValueError(f"{data_path} is missing required key(s): {sorted(missing)}")

    names = config["names"]
    n_classes = len(names)
    if n_classes == 0:
        raise ValueError(f"{data_path} 'names' is empty -- no classes to train on")

    # Resolve split paths relative to data.yaml's own directory, same as ultralytics does.
    base_dir = Path(config.get("path", data_path.parent))
    if not Path(base_dir).is_absolute():
        base_dir = data_path.parent / base_dir
    for split in ("train", "val"):
        split_path = base_dir / config[split]
        if not split_path.exists():
            raise FileNotFoundError(f"{data_path} '{split}' path does not exist: {split_path}")
        _check_labels_resolvable(split, split_path)

    return config


def train_yolo(
    data: str | Path,
    model: str = "yolov8s.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/train",
    name: str = "avos_yolov8",
    seed: int = 0,
    **train_kwargs,
) -> Path:
    """Runs ultralytics training. `model` should be a COCO-pretrained
    checkpoint name/path (e.g. yolov8s.pt) -- ultralytics downloads it
    automatically on first use if it's not already local.

    Returns the path to the best checkpoint, read directly from the
    trainer rather than reconstructed from `project`/`name`: ultralytics
    prepends its own task subfolder (e.g. "runs/detect") in front of
    whatever `project` you pass, and auto-increments `name` (e.g.
    "avos_yolov8" -> "avos_yolov8-2") if that run folder already exists
    from a previous attempt -- so the actual save path frequently does not
    match `Path(project) / name`.
    """
    from ultralytics import YOLO

    yolo = YOLO(model)
    yolo.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        seed=seed,
        **train_kwargs,
    )
    return Path(yolo.trainer.best)


def find_best_checkpoint(project: str | Path, name: str) -> Path:
    """Locates a checkpoint from a *known, exact* project/name (e.g. one you
    already confirmed via a prior training run's actual save_dir). Does NOT
    account for ultralytics' task-subfolder prefixing or name
    auto-incrementing -- prefer using the path `train_yolo` returns instead
    of calling this right after training.
    """
    best = Path(project) / name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"expected trained checkpoint at {best}, but it doesn't exist")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Path to your YOLO dataset config (data.yaml)")
    parser.add_argument("--model", default="yolov8s.pt", help="COCO-pretrained starting checkpoint (default: yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/train", help="ultralytics output directory root")
    parser.add_argument("--name", default="avos_yolov8", help="ultralytics run name (subfolder under --project)")
    parser.add_argument(
        "--out", default="models/yolov8_avos_best.pt", help="Where to copy the best checkpoint after training"
    )
    args = parser.parse_args()

    config = validate_data_yaml(args.data)
    print(f"Training on {len(config['names'])} classes: {config['names']}")

    best = train_yolo(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out_path)
    print(f"Copied best checkpoint {best} -> {out_path}")
    print(f"Use model_path: {out_path} in configs/stage1_datasets.yaml for the Stage 1 evaluation.")


if __name__ == "__main__":
    main()
