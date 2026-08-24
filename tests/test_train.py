import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stage1_detection.train import find_best_checkpoint, validate_data_yaml  # noqa: E402


def _make_split(tmp_path: Path) -> Path:
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)
    return tmp_path


def test_validate_data_yaml_accepts_well_formed_config(tmp_path):
    _make_split(tmp_path)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.dump(
            {
                "train": "images/train",
                "val": "images/val",
                "names": ["bovie", "needle_driver", "forceps"],
            }
        )
    )
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
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.dump({"train": "images/train", "val": "images/val", "names": []})
    )
    with pytest.raises(ValueError, match="no classes"):
        validate_data_yaml(data_yaml)


def test_validate_data_yaml_missing_split_path_raises(tmp_path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    # 'val' split directory intentionally not created
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.dump({"train": "images/train", "val": "images/val", "names": ["bovie"]})
    )
    with pytest.raises(FileNotFoundError, match="'val' path does not exist"):
        validate_data_yaml(data_yaml)


def test_validate_data_yaml_respects_explicit_base_path(tmp_path):
    dataset_root = tmp_path / "avos_dataset"
    (dataset_root / "images" / "train").mkdir(parents=True)
    (dataset_root / "images" / "val").mkdir(parents=True)
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
