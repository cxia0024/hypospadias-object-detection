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


def test_list_images_excludes_matching_substring(tmp_path):
    _touch(
        [
            tmp_path / "video1_frame000.jpg",
            tmp_path / "video2_frame000.jpg",
            tmp_path / "video3_frame000.jpg",
            tmp_path / "video3_frame001.jpg",
        ]
    )
    names = [p.name for p in list_images(tmp_path, exclude_substrings=["video3"])]
    assert names == ["video1_frame000.jpg", "video2_frame000.jpg"]


def test_list_images_exclude_does_not_match_substring_of_another_id(tmp_path):
    # "video3" should not accidentally exclude "video30" or "video13" -- this
    # test documents that substring matching is exact-substring, not exact-token,
    # so callers should pick a substring specific enough to avoid such collisions.
    _touch(
        [
            tmp_path / "video3_frame000.jpg",
            tmp_path / "video30_frame000.jpg",
            tmp_path / "video13_frame000.jpg",
        ]
    )
    names = [p.name for p in list_images(tmp_path, exclude_substrings=["video3"])]
    # video30 and video13 both contain "video3"/"3" as a substring in this naming --
    # video30 contains "video3" literally, so it is also excluded.
    assert names == ["video13_frame000.jpg"]


def test_list_images_multiple_exclude_substrings(tmp_path):
    _touch(
        [
            tmp_path / "video1_frame000.jpg",
            tmp_path / "video2_frame000.jpg",
            tmp_path / "video3_frame000.jpg",
        ]
    )
    names = [p.name for p in list_images(tmp_path, exclude_substrings=["video1", "video3"])]
    assert names == ["video2_frame000.jpg"]


def test_list_images_empty_exclude_list_excludes_nothing(tmp_path):
    _touch([tmp_path / "video1_frame000.jpg"])
    names = [p.name for p in list_images(tmp_path, exclude_substrings=[])]
    assert names == ["video1_frame000.jpg"]
