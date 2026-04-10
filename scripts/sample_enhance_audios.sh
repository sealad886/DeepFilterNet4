#!/usr/bin/env zsh

if [[ ! -d $VIRTUAL_ENV ]]; then
  echo "You must run this script in a virtual environment with the DeepFilterNet4 repo installed."
  echo "An example of a comprehensive setup script can be executed with:"
  echo "  ./setup.sh --all --venv /path/to/venv"
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <checkpoint_dir_or_name> [checkpoint_dir_or_name ...]"
  echo ""
  echo "Enhances audio files in ~/DataDump/to-clean using DFNet4-MLX."
  echo "Accepts one or more checkpoint directories (absolute paths or names"
  echo "under ~/DataDump/checkpoints)."
  echo ""
  echo "Examples:"
  echo "  $0 contrastive_awesome_full_vadlite"
  echo "  $0 ckpt_A ckpt_B ckpt_C"
  echo "  $0 /absolute/path/to/checkpoint"
  exit 1
fi

# Build --checkpoint-dir flags for each argument
CKPT_FLAGS=()
for arg in "$@"; do
  # Validate: must be a directory (absolute) or resolvable under fallback base
  if [[ -d "$arg" ]]; then
    CKPT_FLAGS+=(--checkpoint-dir "$arg")
  elif [[ -d "$HOME/DataDump/checkpoints/$arg" ]]; then
    CKPT_FLAGS+=(--checkpoint-dir "$arg")
  else
    echo "ERROR: Invalid checkpoint directory or name: $arg"
    echo "  (looked at: $arg, ~/DataDump/checkpoints/$arg)"
    exit 2
  fi
done

python scripts/fast_enhance.py \
  -i /Users/andrew/DataDump/to-clean \
  -i /Users/andrew/DataDump/to-clean/news \
  --output-base /Users/andrew/DataDump/to-listen \
  "${CKPT_FLAGS[@]}"
