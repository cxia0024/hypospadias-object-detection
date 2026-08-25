import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from instrument_classifier.train import (  # noqa: E402
    list_class_folders,
    split_classification_folders,
    validate_classification_data,
)


def _make_unsplit_dataset(root: Path, counts: dict[str, int]) -> Path:
    for class_name, n in counts.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for i in range(n):
            (class_dir / f"img{i}.jpg").write_bytes(b"fake")
    return root


def test_list_class_folders(tmp_path):
    _make_unsplit_dataset(tmp_path, {"bovie": 3, "forceps": 2})
    (tmp_path / "not_a_dir.txt").write_text("x")
    assert [p.name for p in list_class_folders(tmp_path)] == ["bovie", "forceps"]


def test_split_is_stratified_per_class(tmp_path):
    source = _make_unsplit_dataset(tmp_path / "source", {"bovie": 100, "forceps": 5})
    out_root = tmp_path / "split"
    split_classification_folders(source, out_root, val_fraction=0.2, seed=42)

    bovie_train = list((out_root / "train" / "bovie").iterdir())
    bovie_val = list((out_root / "val" / "bovie").iterdir())
    forceps_train = list((out_root / "train" / "forceps").iterdir())
    forceps_val = list((out_root / "val" / "forceps").iterdir())

    assert len(bovie_train) + len(bovie_val) == 100
    assert len(bovie_val) == 20  # exactly 20% for a class with 100 images
    assert len(forceps_train) + len(forceps_val) == 5
    assert len(forceps_val) >= 1  # rare class still gets at least one val image


def test_split_no_image_appears_in_both_train_and_val(tmp_path):
    source = _make_unsplit_dataset(tmp_path / "source", {"bovie": 20})
    out_root = tmp_path / "split"
    split_classification_folders(source, out_root, val_fraction=0.3, seed=1)

    train_names = {p.name for p in (out_root / "train" / "bovie").iterdir()}
    val_names = {p.name for p in (out_root / "val" / "bovie").iterdir()}
    assert train_names.isdisjoint(val_names)
    assert train_names | val_names == {f"img{i}.jpg" for i in range(20)}


def test_split_reproducible_with_seed(tmp_path):
    source = _make_unsplit_dataset(tmp_path / "source", {"bovie": 20})
    out_a = tmp_path / "split_a"
    out_b = tmp_path / "split_b"
    split_classification_folders(source, out_a, val_fraction=0.3, seed=7)
    split_classification_folders(source, out_b, val_fraction=0.3, seed=7)

    val_a = {p.name for p in (out_a / "val" / "bovie").iterdir()}
    val_b = {p.name for p in (out_b / "val" / "bovie").iterdir()}
    assert val_a == val_b


def test_split_empty_class_folder_raises(tmp_path):
    source = tmp_path / "source"
    (source / "bovie").mkdir(parents=True)
    with pytest.raises(ValueError, match="no images"):
        split_classification_folders(source, tmp_path / "out")


def test_split_no_class_folders_raises(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(FileNotFoundError, match="no class subfolders"):
        split_classification_folders(source, tmp_path / "out")


def test_validate_classification_data_accepts_well_formed(tmp_path):
    for split, n in (("train", 5), ("val", 2)):
        d = tmp_path / split / "bovie"
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"img{i}.jpg").write_bytes(b"fake")
    classes = validate_classification_data(tmp_path)
    assert classes == {"train": ["bovie"], "val": ["bovie"]}


def test_validate_classification_data_missing_val_raises(tmp_path):
    d = tmp_path / "train" / "bovie"
    d.mkdir(parents=True)
    (d / "img0.jpg").write_bytes(b"fake")
    with pytest.raises(FileNotFoundError, match="'val' folder not found"):
        validate_classification_data(tmp_path)


def test_validate_classification_data_empty_class_folder_raises(tmp_path):
    for split in ("train", "val"):
        d = tmp_path / split / "bovie"
        d.mkdir(parents=True)
        if split == "train":
            (d / "img0.jpg").write_bytes(b"fake")
        # val/bovie left empty
    with pytest.raises(ValueError, match="no images"):
        validate_classification_data(tmp_path)


def test_validate_classification_data_mismatched_classes_raises(tmp_path):
    train_dir = tmp_path / "train" / "bovie"
    train_dir.mkdir(parents=True)
    (train_dir / "img0.jpg").write_bytes(b"fake")

    val_dir = tmp_path / "val" / "forceps"
    val_dir.mkdir(parents=True)
    (val_dir / "img0.jpg").write_bytes(b"fake")

    with pytest.raises(ValueError, match="train/val class sets differ"):
        validate_classification_data(tmp_path)


def test_split_then_validate_end_to_end(tmp_path):
    source = _make_unsplit_dataset(tmp_path / "source", {"bovie": 10, "forceps": 8, "needledriver": 3})
    out_root = tmp_path / "split"
    split_classification_folders(source, out_root, val_fraction=0.25, seed=0)
    classes = validate_classification_data(out_root)
    assert classes["train"] == classes["val"] == ["bovie", "forceps", "needledriver"]
