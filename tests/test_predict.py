import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stage1_detection.predict import list_images  # noqa: E402


def _touch(paths):
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake")


def test_list_images_no_filter(tmp_path):
    _touch(
        [
            tmp_path / "video1_frame000.jpg",
            tmp_path / "video2_frame000.jpg",
            tmp_path / "video3_frame000.jpg",
        ]
    )
    names = [p.name for p in list_images(tmp_path)]
    assert names == ["video1_frame000.jpg", "video2_frame000.jpg", "video3_frame000.jpg"]


def test_list_images_excludes_matching_prefix(tmp_path):
    _touch(
        [
            tmp_path / "video1_frame000.jpg",
            tmp_path / "video2_frame000.jpg",
            tmp_path / "video3_frame000.jpg",
            tmp_path / "video3_frame001.jpg",
        ]
    )
    names = [p.name for p in list_images(tmp_path, exclude_prefixes=["video3"])]
    assert names == ["video1_frame000.jpg", "video2_frame000.jpg"]


def test_list_images_prefix_with_separator_avoids_collision(tmp_path):
    # Excluding "3_" (with the separator) must NOT also drop "30_..." or
    # "13_..." -- this is exactly why exclusion is prefix-match (startswith),
    # not "contains anywhere": a bare "3" substring would incorrectly match
    # "13_frame000.jpg" too, but "3_" as a prefix only matches filenames that
    # actually start with "3_".
    _touch(
        [
            tmp_path / "3_frame000.jpg",
            tmp_path / "30_frame000.jpg",
            tmp_path / "13_frame000.jpg",
        ]
    )
    names = [p.name for p in list_images(tmp_path, exclude_prefixes=["3_"])]
    assert names == ["13_frame000.jpg", "30_frame000.jpg"]


def test_list_images_bare_prefix_without_separator_can_still_collide(tmp_path):
    # Documents the remaining edge case: a prefix without a separator ("3")
    # still prefix-matches "30_...", since "30_..." does start with "3".
    # Callers should include the separator in the prefix to avoid this.
    _touch(
        [
            tmp_path / "3_frame000.jpg",
            tmp_path / "30_frame000.jpg",
        ]
    )
    names = [p.name for p in list_images(tmp_path, exclude_prefixes=["3"])]
    assert names == []


def test_list_images_multiple_exclude_prefixes(tmp_path):
    _touch(
        [
            tmp_path / "video1_frame000.jpg",
            tmp_path / "video2_frame000.jpg",
            tmp_path / "video3_frame000.jpg",
        ]
    )
    names = [p.name for p in list_images(tmp_path, exclude_prefixes=["video1", "video3"])]
    assert names == ["video2_frame000.jpg"]


def test_list_images_empty_exclude_list_excludes_nothing(tmp_path):
    _touch([tmp_path / "video1_frame000.jpg"])
    names = [p.name for p in list_images(tmp_path, exclude_prefixes=[])]
    assert names == ["video1_frame000.jpg"]
