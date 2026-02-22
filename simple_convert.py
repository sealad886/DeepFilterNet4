#!/usr/bin/env python3
"""
Simple script to convert GGUF to MLX using mlx_lm tools
"""

import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    # Download the specific GGUF file
    repo_id = "TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-GGUF"
    filename = "glm-4.7-flash-claude-4.5-opus.bf16.gguf"

    print(f"Downloading {filename} from {repo_id}...")
    local_dir = "./models/gguf"
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    downloaded_file = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

    print(f"File downloaded to: {downloaded_file}")

    # Now try to convert using mlx_lm
    output_dir = "./models/mlx_converted"

    # Remove output directory if it exists
    if Path(output_dir).exists():
        import shutil

        shutil.rmtree(output_dir)

    # Try to use mlx_lm to convert
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.convert",
        "--hf-path",
        downloaded_file,  # Using the downloaded GGUF file
        "--mlx-path",
        output_dir,
        "--dtype",
        "bfloat16",
    ]

    print(f"Attempting conversion with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("Conversion successful!")
        print(result.stdout)

        # Now upload to Hugging Face Hub
        hf_token = os.environ.get("HF_TOKEN_WRITE")
        if not hf_token:
            print("HF_TOKEN_WRITE environment variable not set, skipping upload")
            return

        repo_id_upload = "sealad886/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-mlx"

        # Use huggingface_hub to upload
        from huggingface_hub import HfApi

        api = HfApi()

        print(f"Uploading to {repo_id_upload}...")
        api.upload_folder(folder_path=output_dir, repo_id=repo_id_upload, repo_type="model", token=hf_token)

        print(f"Model uploaded successfully to: https://huggingface.co/{repo_id_upload}")
    else:
        print(f"Conversion failed with error: {result.stderr}")
        # Note: mlx_lm may not support direct GGUF to MLX conversion


if __name__ == "__main__":
    main()
