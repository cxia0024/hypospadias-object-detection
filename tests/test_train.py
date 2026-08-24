import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stage1_detection.train import (  # noqa: E402
    find_best_checkpoint,
    image_to_label_path,
    validate_data_yaml,
)


def _make_split(tmp_path: Path, images_dirname: str = "images", labels_dirname: str = "labels") -> Path:
    """A minimal but real train/val split: one dummy image + matching label per split."""
    for split in ("train", "val"):
        img_dir = tmp_path / images_dirname / split
        lbl_dir = tmp_path / labels_dirname / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        (img_dir / f"{split}_frame0.jpg").write_bytes(b"fake jpg bytes")
        (lbl_dir / f"{split}_frame0.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return tmp_path


def _write_data_yaml(tmp_path: Path, images_dirname: str = "images", names=("bovie", "needle_driver", "forceps")) -> Path:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.dump(
            {
                "train": f"{images_dirname}/train",
                "val": f"{images_dirname}/val",
                "names": list(names),
            }
        )
    )
    return data_yaml


def test_image_to_label_path_standard_layout():
    img = Path("/data/avos/images/train/case001_frame0.jpg")
    label = image_to_label_path(img)
    assert label == Path("/data/avos/labels/train/case001_frame0.txt")


def test_image_to_label_path_no_images_segment_returns_none():
    # e.g. a folder literally named 'images_split' instead of 'images'
    img = Path("/data/avos/images_split/train/case001_frame0.jpg")
    assert image_to_label_path(img) is None


def test_validate_data_yaml_accepts_well_formed_config(tmp_path):
    _make_split(tmp_path)
    data_yaml = _write_data_yaml(tmp_path)
    config = validate_data_yaml(data_yaml)
    assert config["names"] == ["bovie", "needle_driver", "forceps"]


def test_validate_data_yaml_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_data_yaml(tmp_path / "does_not_exist.yaml")


def test_validate_data_yaml_missing_required_key_raises(tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(yaml.dump({"train": "images/train", "val": "images/val"}))  # no 'names'
    with pytest.raises(ValueError, match="missing required key"):
        validate_data_yaml(data_yaml)


def test_validate_data_yaml_empty_names_raises(tmp_path):
    _make_split(tmp_path)
    data_yaml = _write_data_yaml(tmp_path, names=())
    with pytest.raises(ValueError, match="no classes"):
        validate_data_yaml(data_yaml)


def test_validate_data_yaml_missing_split_path_raises(tmp_path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train" / "x.jpg").write_bytes(b"fake")
    (tmp_path / "labels" / "train" / "x.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    # 'val' split directory intentionally not created
    data_yaml = _write_data_yaml(tmp_path, names=("bovie",))
    with pytest.raises(FileNotFoundError, match="'val' path does not exist"):
        validate_data_yaml(data_yaml)


def test_validate_data_yaml_respects_explicit_base_path(tmp_path):
    dataset_root = tmp_path / "avos_dataset"
    _make_split(dataset_root)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.dump(
            {
                "path": str(dataset_root),
                "train": "images/train",
                "val": "images/val",
                "names": ["bovie"],
            }
        )
    )
    config = validate_data_yaml(data_yaml)
    assert config["names"] == ["bovie"]


def test_validate_data_yaml_rejects_non_images_folder_name(tmp_path):
    # Reproduces the real failure: images_split/ + labels_split/ instead of images/ + labels/.
    # Ultralytics can't derive label paths from this and silently treats every image as
    # an unlabeled "background" instead of erroring -- we catch it explicitly here.
    _make_split(tmp_path, images_dirname="images_split", labels_dirname="labels_split")
    data_yaml = _write_data_yaml(tmp_path, images_dirname="images_split")
    with pytest.raises(ValueError, match="images_split"):
        validate_data_yaml(data_yaml)


def test_validate_data_yaml_rejects_missing_matching_labels(tmp_path):
    # Correctly named images/ folder, but no matching labels/ files at all.
    for split in ("train", "val"):
        img_dir = tmp_path / "images" / split
        img_dir.mkdir(parents=True)
        (img_dir / f"{split}_frame0.jpg").write_bytes(b"fake")
        # note: no corresponding labels/ directory created at all
    data_yaml = _write_data_yaml(tmp_path)
    with pytest.raises(FileNotFoundError, match="no matching label"):
        validate_data_yaml(data_yaml)


def test_validate_data_yaml_rejects_split_with_no_images(tmp_path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)
    data_yaml = _write_data_yaml(tmp_path)
    with pytest.raises(FileNotFoundError, match="no image files"):
        validate_data_yaml(data_yaml)


def test_find_best_checkpoint_found(tmp_path):
    weights_dir = tmp_path / "runs" / "avos_yolov8" / "weights"
    weights_dir.mkdir(parents=True)
    best = weights_dir / "best.pt"
    best.write_bytes(b"fake checkpoint")

    found = find_best_checkpoint(tmp_path / "runs", "avos_yolov8")
    assert found == best


def test_find_best_checkpoint_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_best_checkpoint(tmp_path / "runs", "avos_yolov8")
