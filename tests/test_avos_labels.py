import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stage1_detection.avos_labels import read_yolo_label_classes, yolo_split_to_presence_csv  # noqa: E402


def test_read_yolo_label_classes_missing_file_is_empty(tmp_path):
    assert read_yolo_label_classes(tmp_path / "nope.txt") == set()


def test_read_yolo_label_classes_parses_indices(tmp_path):
    label = tmp_path / "x.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.05 0.05\n0 0.9 0.9 0.1 0.1\n")
    assert read_yolo_label_classes(label) == {0, 2}


def test_read_yolo_label_classes_ignores_blank_lines(tmp_path):
    label = tmp_path / "x.txt"
    label.write_text("1 0.5 0.5 0.2 0.2\n\n\n")
    assert read_yolo_label_classes(label) == {1}


def test_read_yolo_label_classes_empty_file_is_empty(tmp_path):
    label = tmp_path / "x.txt"
    label.write_text("")
    assert read_yolo_label_classes(label) == set()


def test_yolo_split_to_presence_csv(tmp_path):
    images_dir = tmp_path / "images" / "val"
    labels_dir = tmp_path / "labels" / "val"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # frame0: bovie (0) and forceps (2) present
    (images_dir / "frame0.jpg").write_bytes(b"fake")
    (labels_dir / "frame0.txt").write_text("0 0.5 0.5 0.1 0.1\n2 0.4 0.4 0.1 0.1\n")

    # frame1: no annotation file at all -> background, everything absent
    (images_dir / "frame1.jpg").write_bytes(b"fake")

    classes = ["bovie", "needledriver", "forceps"]
    out_csv = tmp_path / "labels.csv"
    yolo_split_to_presence_csv(images_dir, labels_dir, classes, out_csv)

    with open(out_csv) as f:
        rows = list(csv.DictReader(f))

    by_frame = {}
    for row in rows:
        by_frame.setdefault(row["frame_id"], {})[row["class"]] = int(row["label"])

    assert by_frame["frame0"] == {"bovie": 1, "needledriver": 0, "forceps": 1}
    assert by_frame["frame1"] == {"bovie": 0, "needledriver": 0, "forceps": 0}


def test_yolo_split_to_presence_csv_output_is_loadable_as_expert_labels(tmp_path):
    from stage1_detection.predict import load_expert_labels

    images_dir = tmp_path / "images" / "val"
    labels_dir = tmp_path / "labels" / "val"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (images_dir / "frame0.jpg").write_bytes(b"fake")
    (labels_dir / "frame0.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    out_csv = tmp_path / "labels.csv"
    yolo_split_to_presence_csv(images_dir, labels_dir, ["bovie"], out_csv)

    df = load_expert_labels(out_csv)  # must not raise "unfinished labeling template"
    assert df.loc[0, "label"] == 1
