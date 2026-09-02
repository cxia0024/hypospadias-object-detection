"""Derive frame-level presence/absence ground truth from AVOS's own YOLO
bounding-box label .txt files.

Unlike hypospadias_eval (raw video that needs fresh expert labeling), the
AVOS val split's .txt files ARE the original expert annotations -- this just
reduces "which boxes are in this image" down to "which classes are present
in this image" (the unit the Stage 1 eval is scored on), with no new
labeling step required.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .predict import get_model_classes, list_images


def read_yolo_label_classes(label_path: Path) -> set[int]:
    """Class indices present in one YOLO label .txt file.

    A missing file means no objects were annotated in that image -- a
    legitimate "background" image in YOLO convention, not a data error.
    """
    if not label_path.exists():
        return set()
    indices = set()
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        indices.add(int(line.split()[0]))
    return indices


def yolo_split_to_presence_csv(
    images_dir: str | Path,
    labels_dir: str | Path,
    classes: list[str],
    out_csv: str | Path,
    exclude_prefixes: list[str] | None = None,
) -> Path:
    """Writes a `frame_id,class,label` CSV (the format `load_expert_labels`
    expects) from a YOLO images/ + labels/ split. `classes` must be in the
    same index order the .txt files were annotated with -- i.e. the `names:`
    list from the data.yaml used to create them (get_model_classes on a
    checkpoint trained on that same data.yaml gives the same order).

    `exclude_prefixes` drops any frame whose filename stem *starts with* one
    of the given prefixes -- e.g. to remove a specific source video from the
    evaluation without touching the underlying image/label files. Include
    the separator (e.g. "3_" not bare "3") if filenames look like
    "3_frame001.jpg", so excluding video 3 doesn't also catch video 30.
    Pass the same list to `predict_presence` too so inference isn't
    wastefully run on frames that will just be dropped anyway.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for img_path in list_images(images_dir, exclude_prefixes=exclude_prefixes):
        present_indices = read_yolo_label_classes(labels_dir / f"{img_path.stem}.txt")
        for idx, cls in enumerate(classes):
            rows.append([img_path.stem, cls, int(idx in present_indices)])

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "class", "label"])
        writer.writerows(rows)

    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_dir", required=True, help="AVOS split images folder, e.g. images/val")
    parser.add_argument("--labels_dir", required=True, help="Matching YOLO label .txt folder, e.g. labels/val")
    parser.add_argument("--model_path", default=None, help="Derive class order from this checkpoint")
    parser.add_argument("--classes", default=None, help="Comma-separated class list, overrides --model_path")
    parser.add_argument("--out_csv", required=True, help="Where to write the frame_id,class,label CSV")
    parser.add_argument(
        "--exclude_frame_id_prefixes",
        default=None,
        help=(
            "Comma-separated prefixes; any frame whose filename starts with one of them is "
            "dropped (e.g. --exclude_frame_id_prefixes 3_ to remove video 3, given filenames "
            "like 3_frame001.jpg -- include the separator so video 3 doesn't also match video 30). "
            "Non-destructive -- the underlying image/label files are untouched."
        ),
    )
    args = parser.parse_args()

    if args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    elif args.model_path:
        classes = get_model_classes(args.model_path)
    else:
        raise SystemExit("must pass --classes or --model_path")

    exclude_prefixes = None
    if args.exclude_frame_id_prefixes:
        exclude_prefixes = [s.strip() for s in args.exclude_frame_id_prefixes.split(",") if s.strip()]

    out = yolo_split_to_presence_csv(
        args.images_dir, args.labels_dir, classes, args.out_csv, exclude_prefixes=exclude_prefixes
    )
    print(f"Wrote presence/absence ground truth for {len(classes)} classes to {out}")


if __name__ == "__main__":
    main()
