#!/usr/bin/env zsh

if [[ ! -d $VIRTUAL_ENV ]]; then
  echo "You must run this script in a virtual environment with the DeepFilterNet4 repo installed."
  echo "An example of a comprehensive setup script can be executed with:"
  echo "  ./setup.sh --all --venv /path/to/venv"
  exit 1
fi

CHECKPOINT_NAME=$1
if [[ ! -d "$CHECKPOINT_NAME" ]]; then
  # first try pre-pending my own regular checkpoint dir
  CHECKPOINT_NAME_UPDATED="$HOME/DataDump/checkpoints/$CHECKPOINT_NAME"
  if [[ ! -d "$CHECKPOINT_NAME_UPDATED" ]]; then
    echo "Invalid path to checkpoint or checkpoint name: $CHECKPOINT_NAME"
    exit 2
  fi
  CHECKPOINT_NAME="$CHECKPOINT_NAME_UPDATED"
fi

python scripts/fast_enhance.py \
  -i /Users/andrew/DataDump/to-clean \
  -o /Users/andrew/DataDump/to-listen/contrastive_awesome_full_vadlite/DeepFilterNet4-MLX \
  -i /Users/andrew/DataDump/to-clean/news \
  -o /Users/andrew/DataDump/to-listen/contrastive_awesome_full_vadlite/news/DeepFilterNet4-MLX \
  --checkpoint-dir "$CHECKPOINT_NAME"
