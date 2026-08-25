"""Train a YOLOv8 image classifier on a folder-per-instrument dataset with
no bounding boxes (e.g. SID-RAS) -- a separate model from the AVOS YOLOv8
*detector* in stage1_detection, since there's no box supervision here to
train a detector on. This is whole-image classification: "which instrument
is the dominant subject of this photo", not "where in this frame is it".

Two steps:
    1. split  -- one folder per class, all images unsplit -> stratified train/val
    2. train  -- fine-tune a COCO/ImageNet-pretrained yolov8-cls checkpoint

Usage:
    python -m instrument_classifier.train split \
        --source_root path/to/sidras --out_root data/sidras_split --val_fraction 0.2

    python -m instrument_classifier.train train \
        --data data/sidras_split --epochs 100 --out models/sidras_classifier_best.pt
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_class_folders(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_dir())


def _list_images(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def split_classification_folders(
    source_root: str | Path,
    out_root: str | Path,
    val_fraction: float = 0.2,
    seed: int = 42,
    mode: str = "copy",
) -> Path:
    """`source_root` has one subfolder per instrument, all images unsplit
    (e.g. source_root/bovie/*.jpg). Writes out_root/train/<class>/... and
    out_root/val/<class>/... via a stratified per-class random split -- every
    class gets its own val_fraction applied independently, so a rare
    instrument class isn't accidentally left with zero val images by a
    single global shuffle.

    `mode="symlink"` avoids duplicating image bytes on disk; use "copy"
    (default) if the source and destination might be on different
    filesystems/mounts (e.g. Drive) where symlinks don't reliably resolve.
    """
    source_root = Path(source_root)
    out_root = Path(out_root)
    rng = random.Random(seed)

    class_dirs = list_class_folders(source_root)
    if not class_dirs:
        raise FileNotFoundError(f"no class subfolders found under {source_root}")

    for class_dir in class_dirs:
        images = _list_images(class_dir)
        if not images:
            raise ValueError(f"class folder {class_dir} has no images")

        shuffled = images.copy()
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction))
        val_images = shuffled[:n_val]
        train_images = shuffled[n_val:]

        for split_name, split_images in (("train", train_images), ("val", val_images)):
            split_dir = out_root / split_name / class_dir.name
            split_dir.mkdir(parents=True, exist_ok=True)
            for img in split_images:
                dest = split_dir / img.name
                if mode == "symlink":
                    if not dest.exists():
                        dest.symlink_to(img.resolve())
                else:
                    shutil.copy(img, dest)

        print(f"[{class_dir.name}] {len(train_images)} train / {len(val_images)} val (of {len(images)} total)")

    return out_root


def validate_classification_data(root: str | Path) -> dict[str, list[str]]:
    """Checks `root` has train/ and val/ subfolders, each with the same set
    of non-empty class subfolders -- a mismatch here (e.g. a class present
    in train but missing from val) otherwise surfaces as a confusing
    ultralytics error mid-training rather than an immediate, clear one.
    """
    root = Path(root)
    classes_by_split: dict[str, list[str]] = {}

    for split in ("train", "val"):
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"'{split}' folder not found under {root}")

        class_dirs = list_class_folders(split_dir)
        if not class_dirs:
            raise ValueError(f"'{split}' has no class subfolders under {split_dir}")

        empty = [c.name for c in class_dirs if not _list_images(c)]
        if empty:
            raise ValueError(f"'{split}' has class folder(s) with no images: {sorted(empty)}")

        classes_by_split[split] = sorted(c.name for c in class_dirs)

    if classes_by_split["train"] != classes_by_split["val"]:
        only_train = sorted(set(classes_by_split["train"]) - set(classes_by_split["val"]))
        only_val = sorted(set(classes_by_split["val"]) - set(classes_by_split["train"]))
        raise ValueError(
            f"train/val class sets differ under {root} -- "
            f"train only: {only_train or 'none'}, val only: {only_val or 'none'}"
        )

    return classes_by_split


def train_classifier(
    data: str | Path,
    model: str = "yolov8n-cls.pt",
    epochs: int = 100,
    imgsz: int = 224,
    batch: int = 64,
    project: str = "runs/train",
    name: str = "instrument_classifier",
    seed: int = 0,
    **train_kwargs,
) -> Path:
    """`data` is the ROOT folder containing train/ and val/ subfolders (each
    with one subfolder per class) -- ultralytics classification mode takes
    this root directly, not a data.yaml like detection training does.

    `model` defaults to yolov8n-cls.pt (ImageNet-pretrained); ultralytics
    downloads it automatically if not already local. Returns the real best
    checkpoint path read from the trainer -- see stage1_detection/train.py's
    train_yolo for why reconstructing this path from project/name is unreliable
    (ultralytics prepends its own task subfolder and auto-increments the run name).
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split", help="Split one-folder-per-class images into train/val")
    split_parser.add_argument("--source_root", required=True, help="Folder with one subfolder per instrument")
    split_parser.add_argument("--out_root", required=True, help="Where to write train/ and val/ subfolders")
    split_parser.add_argument("--val_fraction", type=float, default=0.2)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--mode", choices=["copy", "symlink"], default="copy")

    train_parser = subparsers.add_parser("train", help="Fine-tune a YOLOv8 classifier")
    train_parser.add_argument("--data", required=True, help="Root folder containing train/ and val/ subfolders")
    train_parser.add_argument("--model", default="yolov8n-cls.pt", help="ImageNet-pretrained starting checkpoint")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--imgsz", type=int, default=224)
    train_parser.add_argument("--batch", type=int, default=64)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--project", default="runs/train")
    train_parser.add_argument("--name", default="instrument_classifier")
    train_parser.add_argument("--out", default="models/instrument_classifier_best.pt")

    args = parser.parse_args()

    if args.command == "split":
        out = split_classification_folders(
            args.source_root, args.out_root, val_fraction=args.val_fraction, seed=args.seed, mode=args.mode
        )
        validate_classification_data(out)
        print(f"Split written to {out}, validated OK.")

    elif args.command == "train":
        classes = validate_classification_data(args.data)
        print(f"Training on {len(classes['train'])} classes: {classes['train']}")

        best = train_classifier(
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


if __name__ == "__main__":
    main()
