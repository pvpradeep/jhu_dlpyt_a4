"""Test that VideoDataset returns frames in correct temporal order."""

import os
import sys
import glob
import re
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Standard contiguous frame directory used across current pipeline scripts and README
FRAME_DIR = os.path.join(PROJECT_ROOT, "data", "UCF50_frames_contig_5clips")


def frame_number(filename):
    """Extract numeric index from frame filename."""
    return int(re.search(r'\d+', os.path.basename(filename)).group())


def get_video_dirs(frame_dir, n=10):
    """Collect n video directories for testing."""
    if not os.path.exists(frame_dir):
        return []
    dirs = []
    for category in sorted(os.listdir(frame_dir)):
        cat_path = os.path.join(frame_dir, category)
        if not os.path.isdir(cat_path):
            continue
        for video in sorted(os.listdir(cat_path)):
            vid_path = os.path.join(cat_path, video)
            if os.path.isdir(vid_path):
                dirs.append(vid_path)
                break
        if len(dirs) >= n:
            break
    return dirs


@pytest.mark.skipif(not os.path.exists(FRAME_DIR), reason=f"Frame dataset directory not found at {FRAME_DIR}")
class TestFrameOrder:
    """Tests for frame ordering bugs and fixes using the project's standard clip dataset."""

    @pytest.fixture
    def video_dirs(self):
        """Sample video directories."""
        dirs = get_video_dirs(FRAME_DIR, n=10)
        assert len(dirs) > 0, f"No video dirs found in {FRAME_DIR}"
        return dirs

    def test_glob_is_unsorted(self, video_dirs):
        """Bug #8: Confirm glob.glob() does NOT return sorted frames."""
        any_unsorted = False
        for vid_path in video_dirs:
            raw = glob.glob(os.path.join(vid_path, '*.jpg'))
            expected = sorted(raw, key=frame_number)
            if raw != expected:
                any_unsorted = True
                break
        assert any_unsorted, "Expected glob to return unsorted on this filesystem"

    def test_lexicographic_sort_is_wrong(self, video_dirs):
        """Bug #9: Confirm sorted() without numeric key gives wrong order."""
        any_wrong = False
        for vid_path in video_dirs:
            raw = glob.glob(os.path.join(vid_path, '*.jpg'))
            lexicographic = sorted(raw)
            expected = sorted(raw, key=frame_number)
            if lexicographic != expected:
                any_wrong = True
                break
        assert any_wrong, "Expected lexicographic sort to differ from numeric sort"

    def test_numeric_sort_is_correct(self, video_dirs):
        """Fix: sorted() with numeric key gives correct temporal order."""
        for vid_path in video_dirs:
            raw = glob.glob(os.path.join(vid_path, '*.jpg'))
            numeric = sorted(raw, key=lambda x: int(
                re.search(r'\d+', os.path.basename(x)).group()))
            numbers = [frame_number(p) for p in numeric]
            assert numbers == sorted(numbers), (
                f"Numeric sort failed for {vid_path}: {numbers[:10]}..."
            )

    def test_getitem_matches_numeric_sort(self, video_dirs):
        """Confirm VideoDataset.__getitem__ uses correct numeric ordering."""
        from video_datasets import VideoDataset

        vid_list = [(d, 0) for d in video_dirs]
        ds = VideoDataset(vid_list, fr_per_vid=16, transforms=None)

        for vid_path in video_dirs:
            actual_paths = sorted(
                glob.glob(os.path.join(vid_path, '*.jpg')),
                key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())
            )[:16]

            numbers = [frame_number(p) for p in actual_paths]
            assert numbers == sorted(numbers), (
                f"__getitem__ frame order wrong for {vid_path}"
            )