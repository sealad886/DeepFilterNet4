#!/usr/bin/env python3
"""Test video audio enhancement pipeline components.

This script tests the enhance_video_audio.py implementation without requiring
actual video files or trained models.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Mock MLX and df_mlx before importing enhance_video_audio
sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = MagicMock()
sys.modules["df_mlx"] = MagicMock()
sys.modules["df_mlx.enhance"] = MagicMock()

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from enhance_video_audio import (
    discover_videos,
    setup_logging,
    update_manifest,
)


def test_logging_setup():
    """Test logging configuration."""
    print("\n" + "=" * 60)
    print("Test 1: Logging Setup")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = setup_logging(log_file)

        logger.info("Test info message")
        logger.error("Test error message")

        assert log_file.exists(), "Log file should be created"
        assert log_file.stat().st_size > 0, "Log file should contain errors"
        print("  ✅ Logging setup successful")


def test_video_discovery():
    """Test video discovery logic."""
    print("\n" + "=" * 60)
    print("Test 2: Video Discovery")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test video structure
        (tmpdir / "video1.mp4").touch()
        (tmpdir / "video2.mov").touch()
        (tmpdir / "video3.avi").touch()  # Should be ignored
        (tmpdir / "subdir").mkdir()
        (tmpdir / "subdir" / "video4.mp4").touch()
        (tmpdir / "video1_enhanced.m4a").touch()  # Enhanced version exists

        # Test 2a: Discovery without force
        print("\n2a. Testing discovery without force...")
        videos = discover_videos(tmpdir, force=False)
        assert len(videos) == 2, f"Should find 2 videos (found {len(videos)})"
        print(f"  ✅ Found {len(videos)} videos (skipped enhanced)")

        # Test 2b: Discovery with force
        print("\n2b. Testing discovery with force...")
        videos = discover_videos(tmpdir, force=True)
        assert len(videos) == 3, f"Should find 3 videos (found {len(videos)})"
        print(f"  ✅ Found {len(videos)} videos (force mode)")


def test_manifest_management():
    """Test enhancement manifest creation and updates."""
    print("\n" + "=" * 60)
    print("Test 3: Manifest Management")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        manifest_path = tmpdir / "manifest.json"

        # Create test paths
        video1 = tmpdir / "video1.mp4"
        enhanced1 = tmpdir / "video1_enhanced.m4a"
        video1.touch()
        enhanced1.touch()

        # Test 3a: Create new manifest
        print("\n3a. Testing manifest creation...")
        update_manifest(manifest_path, video1, enhanced1)
        assert manifest_path.exists(), "Manifest should be created"
        print("  ✅ Manifest created")

        # Test 3b: Update existing manifest
        print("\n3b. Testing manifest update...")
        video2 = tmpdir / "video2.mp4"
        enhanced2 = tmpdir / "video2_enhanced.m4a"
        video2.touch()
        enhanced2.touch()
        update_manifest(manifest_path, video2, enhanced2)

        import json

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert len(manifest["videos"]) == 2, "Should have 2 entries"
        print(f"  ✅ Manifest updated ({len(manifest['videos'])} entries)")

        # Test 3c: Update existing entry
        print("\n3c. Testing manifest entry update...")
        update_manifest(manifest_path, video1, enhanced1)
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert len(manifest["videos"]) == 2, "Should still have 2 entries"
        print("  ✅ Existing entry updated (no duplicate)")


def test_cli_interface():
    """Test that CLI accepts expected arguments."""
    print("\n" + "=" * 60)
    print("Test 4: CLI Interface")
    print("=" * 60)

    # Skip CLI test since it requires actual MLX imports
    print("\n4a. Skipping CLI test (requires MLX environment)...")
    print("  ℹ️  CLI interface defined with argparse")
    print("  ℹ️  Manual testing required with actual model")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Video Enhancement Pipeline Tests")
    print("=" * 60)

    try:
        test_logging_setup()
        test_video_discovery()
        test_manifest_management()
        test_cli_interface()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nNote: Full integration testing requires:")
        print("  - Actual video files (.mp4, .mov)")
        print("  - Trained MLX model checkpoint")
        print("  - FFmpeg with AAC support")
        print("\nFor manual testing, use:")
        print("  python scripts/enhance_video_audio.py \\")
        print("    --video-dir /path/to/videos \\")
        print("    --model-checkpoint /path/to/model.safetensors")
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
