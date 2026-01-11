# Video Audio Enhancement and Playback Integration

Complete guide for using DeepFilterNet4 to enhance video audio and play it back with switchable audio tracks in VLC/MPV.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Training Pipeline](#training-pipeline)
- [Video Enhancement](#video-enhancement)
- [Playback Integration](#playback-integration)
  - [VLC Playback](#vlc-playback)
  - [MPV Playback](#mpv-playback)
  - [Mixed Audio](#mixed-audio-optional)
- [Troubleshooting](#troubleshooting)
- [Examples](#complete-workflow-example)

---

## Overview

This system provides three integrated components:

1. **Training Pipeline**: Automated 100-epoch training for wall and dynamic models
2. **Video Enhancement**: Batch audio extraction, enhancement, and AAC encoding
3. **Playback Integration**: VLC/MPV playlists with switchable original/enhanced audio tracks

---

## Prerequisites

### System Requirements

- macOS (Apple Silicon or Intel)
- Python 3.10+
- FFmpeg with AAC codec support
- VLC and/or MPV media players

### Installation

```bash
# Install FFmpeg (if not already installed)
brew install ffmpeg

# Install VLC
brew install --cask vlc

# Install MPV
brew install mpv

# Install Python dependencies
cd DeepFilterNet
poetry install

# Or if using pip
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check FFmpeg
ffmpeg -version
ffmpeg -encoders | grep -E "(pcm_s16le|aac)"

# Check VLC
/Applications/VLC.app/Contents/MacOS/VLC --version

# Check MPV
mpv --version
```

---

## Training Pipeline

### Automated 100-Epoch Training

The `run_training_pipeline.sh` script orchestrates sequential training of wall and dynamic models with full logging and checkpointing.

#### Usage

```bash
./scripts/run_training_pipeline.sh \
  --wall-dataset /path/to/wall_processed \
  --wall-checkpoint /path/to/checkpoints/dfnetmf_wall \
  --dynamic-cache /path/to/audio_cache \
  --dynamic-checkpoint /path/to/checkpoints/dfnet4_dynamic \
  --epochs 100 \
  --wall-batch-size 32 \
  --dynamic-batch-size 8
```

#### Options

- `--wall-dataset PATH` - Path to wall camera dataset
- `--wall-checkpoint PATH` - Wall training checkpoint directory
- `--dynamic-cache PATH` - Audio cache for dynamic training
- `--dynamic-checkpoint PATH` - Dynamic training checkpoint directory
- `--epochs N` - Number of epochs per training run (default: 100)
- `--wall-batch-size N` - Batch size for wall training (default: 32)
- `--dynamic-batch-size N` - Batch size for dynamic training (default: 8)
- `--wall-method METHOD` - Wall training method: WF or MVDR (default: WF)

#### Features

- **Sequential Execution**: Wall training completes before dynamic training starts
- **Resume Support**: Automatically resumes from latest checkpoint if interrupted
- **Timestamped Logging**: Separate logs for each phase (`logs/wall_training_*.log`, `logs/dynamic_training_*.log`)
- **Signal Handling**: CTRL+C saves checkpoint before exit
- **Duration Tracking**: Reports training time for each phase

#### Outputs

- Wall model checkpoints: `<wall-checkpoint>/step_*.safetensors`
- Dynamic model checkpoints: `<dynamic-checkpoint>/epoch_*.safetensors`
- Training logs: `logs/pipeline_*.log`

---

## Video Enhancement

### Batch Audio Enhancement

The `enhance_video_audio.py` script processes video directories recursively, extracting audio, enhancing it with trained MLX models, and encoding to AAC.

#### Basic Usage

```bash
python scripts/enhance_video_audio.py \
  --video-dir /Volumes/HomeSecurityVideos/Wall \
  --model-checkpoint ./checkpoints/dfnet4_dynamic/best.safetensors
```

#### Advanced Usage

```bash
python scripts/enhance_video_audio.py \
  --video-dir /Volumes/HomeSecurityVideos/Blink \
  --model-checkpoint ./checkpoints/dfnet4_dynamic/best.safetensors \
  --output-format aac \
  --force \
  --manifest-path ./playlists/blink_manifest.json \
  --log-file ./logs/enhancement_errors.log
```

#### Options

- `--video-dir PATH` - Directory containing videos (searched recursively)
- `--model-checkpoint PATH` - Trained MLX model (.safetensors file)
- `--output-format {aac,wav}` - Output format (default: aac)
- `--force` - Re-process videos that already have enhanced audio
- `--manifest-path PATH` - Enhancement manifest location (default: video_dir/enhancement_manifest.json)
- `--log-file PATH` - Error log file (default: errors.log)

#### Supported Video Formats

- `.mp4`
- `.mov`

#### Process

1. **Discovery**: Recursively scans for `.mp4` and `.mov` files
2. **Extraction**: FFmpeg extracts audio to PCM WAV (48kHz, 16-bit)
3. **Enhancement**: MLX model processes audio
4. **Encoding**: FFmpeg encodes to AAC (192k bitrate) or saves as WAV
5. **Manifest**: Updates JSON manifest with video → enhanced audio mappings

#### Outputs

- Enhanced audio: `<video_name>_enhanced.m4a` (or `.wav`)
- Manifest: `enhancement_manifest.json`
- Error log: `errors.log`

#### Performance Notes

- Typical processing time: 1-2 minutes per video (depending on length and hardware)
- Processes one video at a time (batch processing of frames happens internally)
- Uses temporary directory for intermediate WAV files (automatically cleaned up)

---

## Playback Integration

### Generate Playlists

The `generate_playlists.py` script reads the enhancement manifest and generates player-specific playlists.

```bash
python scripts/generate_playlists.py \
  --manifest-path /Volumes/HomeSecurityVideos/Wall/enhancement_manifest.json \
  --player both
```

#### Options

- `--manifest-path PATH` - Enhancement manifest JSON file
- `--output-dir PATH` - Output directory (default: same as manifest)
- `--player {vlc,mpv,both}` - Generate for which player (default: both)
- `--playlist-name NAME` - Base name for playlists (default: enhanced_audio_playlist)
- `--relative-paths` - Use relative paths (not recommended for VLC)
- `--generate-mixed` - Generate pre-mixed audio files
- `--mix-ratios X:Y ...` - Mix ratios (default: 50:50 70:30)

---

### VLC Playback

#### Generated Files

- `enhanced_audio_playlist.m3u` - M3U playlist with sidecar audio directives

#### Usage

1. **Open Playlist in VLC**:
   ```bash
   open -a VLC enhanced_audio_playlist.m3u
   ```

   Or: File → Open File → Select `enhanced_audio_playlist.m3u`

2. **Switch Audio Tracks During Playback**:
   - Menu: **Audio → Audio Track**
   - Select:
     - **Track 1** - Original video audio
     - **Track 2** - Enhanced audio (sidecar)

3. **Keyboard Shortcuts**:
   - `b` - Cycle through audio tracks
   - `Cmd+]` - Next file in playlist
   - `Cmd+[` - Previous file in playlist

#### VLC Configuration Tips

- **Auto-select enhanced audio**: VLC → Preferences → Audio → Preferred audio track (set to 2)
- **Remember position**: VLC → Preferences → Interface → Continue playback? (Ask)

---

### MPV Playback

#### Generated Files

- `enhanced_audio_playlist_mpv.txt` - MPV playlist with audio file comments
- `play_enhanced_audio_playlist_mpv.sh` - Bash script for convenient playback

#### Usage

##### Option 1: MPV Bash Script (Recommended)

```bash
# Play all videos
./play_enhanced_audio_playlist_mpv.sh

# Play specific video (1-based index)
./play_enhanced_audio_playlist_mpv.sh 5
```

##### Option 2: MPV Direct

```bash
mpv --playlist=enhanced_audio_playlist_mpv.txt
```

##### Option 3: Single Video

```bash
mpv video.mp4 --audio-file=video_enhanced.m4a
```

#### Switch Audio Tracks During Playback

- Press `#` to cycle through audio tracks
- Press `_` to cycle backwards through audio tracks

#### MPV Configuration Tips

Create or edit `~/.config/mpv/mpv.conf`:

```conf
# Default to external audio files
audio-file-auto=fuzzy

# Show audio track info on OSD
osd-level=2

# Start with enhanced audio (track 2)
aid=2
```

---

### Mixed Audio (Optional)

Generate pre-mixed audio files blending original and enhanced audio at configurable ratios.

#### Generate Mixed Audio

```bash
python scripts/generate_playlists.py \
  --manifest-path enhancement_manifest.json \
  --generate-mixed \
  --mix-ratios 50:50 70:30 80:20
```

#### Output

- `mixed_audio/<video>_mixed_50-50.m4a` - Equal blend
- `mixed_audio/<video>_mixed_70-30.m4a` - More original
- `mixed_audio/<video>_mixed_80-20.m4a` - Mostly original

#### Usage

Use mixed files as regular video audio replacements:

```bash
# Replace video audio with mixed version
ffmpeg -i video.mp4 -i mixed_50-50.m4a \
  -c:v copy -c:a copy -map 0:v:0 -map 1:a:0 \
  video_remuxed.mp4
```

---

## Troubleshooting

### FFmpeg Errors

#### "Could not find codec 'aac'"

```bash
# Reinstall FFmpeg with AAC support
brew reinstall ffmpeg
```

#### "Invalid duration specification"

- Check that video files are not corrupted
- Try re-encoding problem videos:
  ```bash
  ffmpeg -i broken.mp4 -c copy fixed.mp4
  ```

### Enhancement Errors

#### "Model checkpoint not found"

- Verify checkpoint path is correct
- Ensure model file ends with `.safetensors`
- Check training completed successfully

#### "Audio enhancement failed"

- Check error log: `errors.log`
- Verify input video has audio track:
  ```bash
  ffmpeg -i video.mp4 2>&1 | grep Audio
  ```
- Try extracting audio manually to isolate issue

### Playback Errors

#### VLC: "No audio track found"

- Verify enhanced audio file exists next to video
- Check M3U playlist uses absolute paths (`file://` URLs)
- Try opening video and audio separately first

#### MPV: "Audio file not found"

- Verify paths in MPV script are correct
- Try absolute paths instead of relative
- Run script with `bash -x` for debugging

### Permission Issues

```bash
# Make scripts executable
chmod +x scripts/*.sh
chmod +x scripts/*.py

# Fix audio file permissions
chmod 644 *.m4a
```

---

## Complete Workflow Example

### 1. Train Models (100 epochs each)

```bash
./scripts/run_training_pipeline.sh \
  --epochs 100 \
  --wall-batch-size 32 \
  --dynamic-batch-size 8
```

**Output**: Checkpoints in `checkpoints/dfnetmf_wall/` and `checkpoints/dfnet4_dynamic/`

---

### 2. Enhance Video Audio

```bash
python scripts/enhance_video_audio.py \
  --video-dir /Volumes/HomeSecurityVideos/Wall \
  --model-checkpoint checkpoints/dfnet4_dynamic/best.safetensors \
  --output-format aac
```

**Output**:
- Enhanced audio files: `/Volumes/HomeSecurityVideos/Wall/*_enhanced.m4a`
- Manifest: `/Volumes/HomeSecurityVideos/Wall/enhancement_manifest.json`

---

### 3. Generate Playlists

```bash
python scripts/generate_playlists.py \
  --manifest-path /Volumes/HomeSecurityVideos/Wall/enhancement_manifest.json \
  --player both \
  --generate-mixed \
  --mix-ratios 50:50 70:30
```

**Output**:
- `enhanced_audio_playlist.m3u` (VLC)
- `play_enhanced_audio_playlist_mpv.sh` (MPV)
- `mixed_audio/` directory with pre-mixed tracks

---

### 4. Playback

#### VLC

```bash
open -a VLC enhanced_audio_playlist.m3u
```

During playback: **Audio → Audio Track → Track 2** (enhanced)

#### MPV

```bash
./play_enhanced_audio_playlist_mpv.sh
```

During playback: Press `#` to switch audio tracks

---

## Additional Resources

- **Training logs**: `logs/` directory
- **Enhancement errors**: `errors.log`
- **Pipeline help**: `./scripts/run_training_pipeline.sh --help`
- **Enhancement help**: `python scripts/enhance_video_audio.py --help`
- **Playlist help**: `python scripts/generate_playlists.py --help`

---

## Performance Expectations

### Training

- **Wall training (100 epochs)**: ~12-24 hours (Apple M1/M2, batch size 32)
- **Dynamic training (100 epochs)**: ~18-36 hours (batch size 8)

### Enhancement

- **Per video**: 1-2 minutes (depending on length)
- **100 videos**: ~2-3 hours
- **1000 videos**: ~20-30 hours

### Disk Usage

- **Model checkpoints**: ~10-20 GB
- **Enhanced audio**: ~10-20% of original video sizes
- **Mixed audio (optional)**: ~5-10% of original video sizes per mix ratio

---

## Tips for Best Results

1. **Training**:
   - Use `--resume` flag to continue interrupted training
   - Monitor logs for convergence
   - Keep batch size appropriate for your hardware

2. **Enhancement**:
   - Process smaller batches first to verify quality
   - Use `--force` flag to re-process with new models
   - Monitor error logs for failed videos

3. **Playback**:
   - VLC is more reliable for playlist features
   - MPV script provides better control for sequential viewing
   - Try mixed audio ratios to find your preference

---

## License

See repository LICENSE files for details.
