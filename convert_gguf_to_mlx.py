#!/usr/bin/env python3
"""
Script to convert GGUF model to MLX format and upload to Hugging Face Hub.
"""

import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def install_required_packages():
    """Install required packages if not already installed."""
    required_packages = ["mlx-lm", "gguf"]

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def download_model(repo_id: str = "TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill", file_list: str = ""):
    """Download the original model from Hugging Face Hub (instead of GGUF)."""
    # We'll use the repo_id directly in the conversion step

    # Allow CLI pass-in "repo_id:file1,file2" format for flexibility
    if ":" in repo_id:
        repo_id, file_list = repo_id.split(":", 1)
        file_list = file_list.split(",")
    else:
        file_list = []

    snapshot_download(
        repo_id=repo_id,
    )

    # For conversion with mlx_lm, we can use the repo_id directly
    # Return the repo_id to be used in the conversion
    if file_list:
        return repo_id, file_list
    return repo_id


def convert_model(input_path, from_scratch: bool = True):
    """Convert Hugging Face model to MLX format using mlx_lm."""
    import shutil
    import subprocess
    import sys

    output_dir = Path("~/mlx_models").expanduser().resolve().absolute()

    # Remove the output directory if it exists
    if from_scratch and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting model from {input_path} to MLX format...")

    # Use mlx_lm's convert function to convert the model
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.convert",
        "--hf-path",
        input_path,
        "--mlx-path",
        output_dir,
        "--dtype",
        "bfloat16",  # Match the original .bf16.gguf
    ]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error during conversion: {result.stderr}")
        raise RuntimeError(f"Conversion failed: {result.stderr}")

    print(result.stdout)
    print(f"Model converted to MLX format in: {output_dir}")
    return output_dir


def upload_to_hf(model_path, repo_id, token):
    """Upload the converted model to Hugging Face Hub."""
    api = HfApi()

    print(f"Uploading model to {repo_id}...")

    api.upload_folder(folder_path=model_path, repo_id=repo_id, repo_type="model", token=token)

    print(f"Model uploaded successfully to: https://huggingface.co/{repo_id}")


def main():
    """Main function to orchestrate the conversion and upload process."""
    print("Starting GGUF to MLX conversion process...")

    # Install required packages
    install_required_packages()

    # Download the model
    gguf_path = download_model()

    # Convert to MLX format
    mlx_path = convert_model(gguf_path)

    # Upload to Hugging Face Hub
    hf_token = os.environ.get("HF_TOKEN_WRITE")
    if not hf_token:
        raise ValueError("HF_TOKEN_WRITE environment variable not set")

    repo_id = "sealad886/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-mlx"
    upload_to_hf(mlx_path, repo_id, hf_token)

    print("Conversion and upload completed successfully!")


if __name__ == "__main__":
    main()
