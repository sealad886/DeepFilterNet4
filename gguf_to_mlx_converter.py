#!/usr/bin/env python3
"""
Script to convert GGUF model to MLX format and upload to Hugging Face Hub.
This script manually converts each tensor from GGUF format to MLX format,
then creates the proper MLX model directory structure.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from gguf import GGUFReader
from huggingface_hub import HfApi, hf_hub_download


def install_required_packages():
    """Install required packages if not already installed."""
    required_packages = [("gguf", "gguf"), ("huggingface_hub", "huggingface-hub"), ("mlx", "mlx"), ("mlx_lm", "mlx-lm")]

    import subprocess
    import sys

    for module_name, package_name in required_packages:
        try:
            __import__(module_name.replace("-", "_"))
        except ImportError:
            print(f"Installing {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def download_gguf_file(repo_id, filename, local_dir="./models/gguf"):
    """Download the specific GGUF file from Hugging Face Hub."""
    print(f"Downloading {filename} from {repo_id}...")
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    downloaded_file = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

    print(f"File downloaded to: {downloaded_file}")
    return downloaded_file


def load_gguf_tensors(gguf_path):
    """Load tensors from GGUF file and convert to MLX arrays."""
    print(f"Loading tensors from GGUF file: {gguf_path}")

    reader = GGUFReader(gguf_path)
    tensors = {}

    # Extract tensor info and data
    for tensor in reader.tensors:
        tensor_name = tensor.name
        tensor_data = tensor.data

        print(f"Processing tensor: {tensor_name} with shape {tensor_data.shape} and dtype {tensor_data.dtype}")

        # Convert numpy array to mlx array
        mlx_tensor = mx.array(tensor_data)
        tensors[tensor_name] = mlx_tensor

    print(f"Loaded {len(tensors)} tensors from GGUF file")
    return tensors


def extract_gguf_metadata(gguf_path):
    """Extract metadata from GGUF file."""
    print(f"Extracting metadata from GGUF file: {gguf_path}")

    reader = GGUFReader(gguf_path)
    metadata = {}
    tokenizer_metadata = {}

    # Extract all metadata
    for field in reader.fields.values():
        key = field.name.decode("utf-8")
        if key.startswith("tokenizer."):
            # Store tokenizer metadata separately
            value = field.parts[field.data[0]] if len(field.data) == 1 else [field.parts[i] for i in field.data]
            tokenizer_metadata[key] = value
        else:
            value = field.parts[field.data[0]] if len(field.data) == 1 else [field.parts[i] for i in field.data]
            metadata[key] = value

    print(f"Extracted {len(metadata)} general metadata entries and {len(tokenizer_metadata)} tokenizer entries")
    return metadata, tokenizer_metadata


def create_config_from_gguf_metadata(gguf_metadata):
    """Create a config.json from GGUF metadata."""
    # Map GGUF metadata to HuggingFace config fields
    config = {}

    # Common mappings from GGUF to HF config
    mapping = {
        "general.architecture": "model_type",
        "llama.context_length": "max_position_embeddings",
        "llama.embedding_length": "hidden_size",
        "llama.feed_forward_length": "intermediate_size",
        "llama.block_count": "num_hidden_layers",
        "llama.attention.head_count": "num_attention_heads",
        "llama.attention.head_count_kv": "num_key_value_heads",
        "llama.rope.dimension_count": "rope_dimension",
        "llama.attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "general.name": "model_name",
        "general.description": "model_description",
    }

    for gguf_key, hf_key in mapping.items():
        if gguf_key in gguf_metadata:
            config[hf_key] = gguf_metadata[gguf_key]

    # Set default values if not present
    if "model_type" not in config:
        config["model_type"] = "auto"  # Will be determined later

    if "vocab_size" not in config:
        # This would typically come from tokenizer, but we'll need to handle it
        print("Warning: vocab_size not found in GGUF metadata. This may need to be set manually.")

    # Add other required fields for GLM models if possible
    if "glm" in config.get("model_name", "").lower():
        config["model_type"] = "chatglm"

    return config


def save_mlx_model(tensors, config, output_dir, metadata=None):
    """Save tensors and config in MLX format."""
    print(f"Saving MLX model to: {output_dir}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save tensors in MLX format (using numpy format that MLX can read)
    weights_file = Path(output_dir) / "weights.npz"
    print(f"Saving weights to: {weights_file}")

    # Convert mlx arrays to numpy for saving
    numpy_tensors = {name: np.array(tensor) for name, tensor in tensors.items()}
    np.savez(weights_file, **numpy_tensors)

    # Save config
    config_path = Path(output_dir) / "config.json"
    print(f"Saving config to: {config_path}")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Create a basic model card
    readme_content = f"""# MLX Converted Model
This model was converted from GGUF format to MLX format using gguf_to_mlx_converter.py.

## Original Model
- Repository: `{metadata.get('original_repo', 'Unknown') if metadata else 'Unknown'}`
- File: `{metadata.get('original_file', 'Unknown') if metadata else 'Unknown'}`

## Conversion Details
- Conversion Script: `gguf_to_mlx_converter.py`
- Conversion Date: `{metadata.get('conversion_date', 'Unknown') if metadata else 'Unknown'}`
"""

    readme_path = Path(output_dir) / "README.md"
    print(f"Saving README to: {readme_path}")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"MLX model saved successfully with {len(tensors)} tensors")


def upload_to_hf(model_path, repo_id, token):
    """Upload the converted model to Hugging Face Hub."""
    print(f"Uploading model to {repo_id}...")

    api = HfApi()

    api.upload_folder(folder_path=model_path, repo_id=repo_id, repo_type="model", token=token)

    print(f"Model uploaded successfully to: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Convert GGUF model to MLX format and upload to Hugging Face Hub")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing output directory")
    parser.add_argument(
        "--output-dir", type=str, default="./models/mlx_converted", help="Output directory for MLX model"
    )

    args = parser.parse_args()

    print("Starting GGUF to MLX conversion process...")

    # Install required packages
    install_required_packages()

    # Define model parameters
    repo_id = "TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-GGUF"
    filename = "glm-4.7-flash-claude-4.5-opus.bf16.gguf"
    repo_id_upload = "sealad886/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-mlx"

    # Check if output directory exists
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if args.force:
            print(f"Removing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise ValueError(f"Output directory {output_dir} already exists. Use --force to overwrite.")

    # Download the GGUF file
    gguf_path = download_gguf_file(repo_id, filename)

    # Load tensors from GGUF
    tensors = load_gguf_tensors(gguf_path)

    # Extract metadata from GGUF
    gguf_metadata = extract_gguf_metadata(gguf_path)

    # Create config from metadata
    config = create_config_from_gguf_metadata(gguf_metadata)

    # Add conversion metadata
    conversion_metadata = {
        "original_repo": repo_id,
        "original_file": filename,
        "conversion_date": str(Path(gguf_path).stat().st_mtime),
    }

    # Save MLX model
    save_mlx_model(tensors, config, output_dir, conversion_metadata)

    # Upload to Hugging Face Hub
    hf_token = os.environ.get("HF_TOKEN_WRITE")
    if not hf_token:
        raise ValueError("HF_TOKEN_WRITE environment variable not set")

    upload_to_hf(output_dir, repo_id_upload, hf_token)

    print("Conversion and upload completed successfully!")


if __name__ == "__main__":
    main()
