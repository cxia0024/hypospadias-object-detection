"""Randomly sample frames from a directory of surgical videos.

Used to build the labeled frame sets for Stage 1 (the `images_dir` an
expert then annotates presence/absence over to make `labels_csv`). Sampling
is per-video and reproducible via `--seed`, and a manifest CSV is written
alongside the extracted JPEGs so labeling can start from `frame_id,video_id`
and just add `class,label` columns.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def list_videos(videos_dir: str | Path, video_exts: set[str] = VIDEO_EXTS) -> list[Path]:
    videos_dir = Path(videos_dir)
    return sorted(p for p in videos_dir.iterdir() if p.suffix.lower() in video_exts)


def sample_frame_indices(n_frames_total: int, k: int, rng: random.Random) -> list[int]:
    """k random, distinct frame indices from [0, n_frames_total), sorted.

    Sampling without replacement so no frame is extracted twice; if the video
    is shorter than k frames, every frame is taken instead of erroring.
    """
    k = min(k, n_frames_total)
    if k <= 0:
        return []
    return sorted(rng.sample(range(n_frames_total), k))


def extract_frames_from_video(
    video_path: str | Path, frame_indices: list[int], output_dir: Path, video_id: str
) -> list[tuple[str, str, int, float | None, str]]:
    """Seeks to each index and writes a JPEG. Returns manifest rows:
    (frame_id, video_id, frame_index, timestamp_sec, output_path).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    rows = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_id = f"{video_id}_frame{idx:06d}"
        out_path = output_dir / f"{frame_id}.jpg"
        cv2.imwrite(str(out_path), frame)
        timestamp = (idx / fps) if fps > 0 else None
        rows.append((frame_id, video_id, idx, timestamp, str(out_path)))
    cap.release()
    return rows


def extract_random_frames(
    videos_dir: str | Path,
    output_dir: str | Path,
    n_per_video: int,
    seed: int = 42,
    video_exts: set[str] = VIDEO_EXTS,
) -> list[tuple[str, str, int, float | None, str]]:
    videos_dir = Path(videos_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    manifest_rows = []
    for video_path in list_videos(videos_dir, video_exts):
        video_id = video_path.stem
        cap = cv2.VideoCapture(str(video_path))
        n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n_frames_total <= 0:
            print(f"[{video_id}] skipped: could not read frame count")
            continue

        frame_indices = sample_frame_indices(n_frames_total, n_per_video, rng)
        rows = extract_frames_from_video(video_path, frame_indices, output_dir, video_id)
        manifest_rows.extend(rows)
        print(f"[{video_id}] extracted {len(rows)}/{len(frame_indices)} requested frames "
              f"(video has {n_frames_total} total)")

    return manifest_rows


def write_manifest(rows: list[tuple[str, str, int, float | None, str]], manifest_path: str | Path) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "video_id", "frame_index", "timestamp_sec", "path"])
        writer.writerows(rows)


UNLABELED_SENTINEL = ""  # empty label cell = "not yet expert-reviewed"


def write_labeling_template(
    rows: list[tuple[str, str, int, float | None, str]], classes: list[str], template_path: str | Path
) -> None:
    """Writes a `frame_id,video_id,class,label` template with every `label`
    cell left blank -- extracted frames are NOT expert-labeled yet. An expert
    must fill in 0/1 per class per frame before this file can be used as a
    dataset's `labels_csv` in configs/stage1_datasets.yaml. `load_expert_labels`
    (predict.py) refuses to load a file with any blank/non-0-1 label cells,
    so an unfinished template can't silently be scored as if it were real ground truth.
    """
    template_path = Path(template_path)
    template_path.parent.mkdir(parents=True, exist_ok=True)
    with open(template_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "video_id", "class", "label"])
        for frame_id, video_id, _frame_index, _timestamp, _path in rows:
            for cls in classes:
                writer.writerow([frame_id, video_id, cls, UNLABELED_SENTINEL])
    print(
        f"Wrote UNLABELED template to {template_path} "
        f"({len(rows)} frames x {len(classes)} classes = {len(rows) * len(classes)} rows). "
        "An expert must fill in the 'label' column (0/1) before this can be used as labels_csv."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos_dir", required=True, help="Directory of source video files")
    parser.add_argument("--output_dir", required=True, help="Directory to write extracted JPEG frames to")
    parser.add_argument("--n_per_video", type=int, required=True, help="Frames to randomly sample per video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, for reproducible sampling")
    parser.add_argument(
        "--manifest", default=None, help="Optional CSV manifest path (defaults to <output_dir>/manifest.csv)"
    )
    parser.add_argument(
        "--classes",
        default=None,
        help=(
            "Comma-separated class list, e.g. bovie,needle_driver,forceps. "
            "If given, also writes an UNLABELED labeling template (labels_template.csv) "
            "for an expert to fill in presence/absence per class per frame."
        ),
    )
    parser.add_argument(
        "--labeling_template",
        default=None,
        help="Optional path for the labeling template (defaults to <output_dir>/labels_template.csv)",
    )
    args = parser.parse_args()

    rows = extract_random_frames(args.videos_dir, args.output_dir, args.n_per_video, seed=args.seed)
    manifest_path = args.manifest or (Path(args.output_dir) / "manifest.csv")
    write_manifest(rows, manifest_path)
    print(f"Extracted {len(rows)} frames total. Manifest: {manifest_path}")

    if args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
        template_path = args.labeling_template or (Path(args.output_dir) / "labels_template.csv")
        write_labeling_template(rows, classes, template_path)


if __name__ == "__main__":
    main()
