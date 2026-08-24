import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stage1_detection.extract_frames import (  # noqa: E402
    extract_random_frames,
    sample_frame_indices,
    write_labeling_template,
)


def test_sample_frame_indices_is_reproducible_with_seed():
    a = sample_frame_indices(1000, 20, random.Random(42))
    b = sample_frame_indices(1000, 20, random.Random(42))
    assert a == b


def test_sample_frame_indices_no_duplicates_and_sorted():
    indices = sample_frame_indices(1000, 20, random.Random(0))
    assert len(indices) == len(set(indices))
    assert indices == sorted(indices)


def test_sample_frame_indices_caps_at_video_length():
    indices = sample_frame_indices(5, 20, random.Random(0))
    assert len(indices) == 5
    assert set(indices) == set(range(5))


def test_sample_frame_indices_zero_frames():
    assert sample_frame_indices(0, 10, random.Random(0)) == []


def _write_synthetic_video(path: Path, n_frames: int, fps: int = 10, size: tuple[int, int] = (32, 32)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), fill_value=i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_extract_random_frames_end_to_end(tmp_path):
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    _write_synthetic_video(videos_dir / "case001.mp4", n_frames=50)
    _write_synthetic_video(videos_dir / "case002.mp4", n_frames=50)

    output_dir = tmp_path / "frames"
    rows = extract_random_frames(videos_dir, output_dir, n_per_video=5, seed=1)

    assert len(rows) == 10  # 5 per video x 2 videos
    video_ids = {r[1] for r in rows}
    assert video_ids == {"case001", "case002"}
    for _frame_id, _video_id, _idx, _ts, path in rows:
        assert Path(path).exists()


def test_extract_random_frames_reproducible_across_runs(tmp_path):
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    _write_synthetic_video(videos_dir / "case001.mp4", n_frames=50)

    rows_a = extract_random_frames(videos_dir, tmp_path / "run_a", n_per_video=5, seed=7)
    rows_b = extract_random_frames(videos_dir, tmp_path / "run_b", n_per_video=5, seed=7)

    indices_a = [r[2] for r in rows_a]
    indices_b = [r[2] for r in rows_b]
    assert indices_a == indices_b


def test_labeling_template_has_blank_labels(tmp_path):
    rows = [("case001_frame000010", "case001", 10, 1.0, "irrelevant.jpg")]
    template_path = tmp_path / "labels_template.csv"
    write_labeling_template(rows, classes=["bovie", "forceps"], template_path=template_path)

    content = template_path.read_text().strip().splitlines()
    assert content[0] == "frame_id,video_id,class,label"
    assert len(content) == 3  # header + 2 classes
    for line in content[1:]:
        assert line.endswith(",")  # label cell is blank


def test_load_expert_labels_rejects_unfilled_template(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from stage1_detection.predict import load_expert_labels

    rows = [("case001_frame000010", "case001", 10, 1.0, "irrelevant.jpg")]
    template_path = tmp_path / "labels_template.csv"
    write_labeling_template(rows, classes=["bovie"], template_path=template_path)

    with pytest.raises(ValueError, match="unfinished labeling template"):
        load_expert_labels(template_path)


def test_load_expert_labels_accepts_filled_template(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from stage1_detection.predict import load_expert_labels

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("frame_id,video_id,class,label\ncase001_frame000010,case001,bovie,1\n")

    df = load_expert_labels(labels_csv)
    assert df.loc[0, "label"] == 1
