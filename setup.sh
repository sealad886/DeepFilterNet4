#!/usr/bin/env bash
# One-step builder for DeepFilterNet4 (Python + Cargo + optional maturin bindings).
# Defaults:
#   - Python: create .venv with python3.10 and install .[asr-mlx]
#   - Cargo:  cargo build --workspace --release --all-features
# Use flags below to toggle extras or skip parts.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ------------------------- defaults ------------------------- #
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
DEFAULT_EXTRAS=("asr-mlx")
USER_EXTRAS=()
BUILD_PYTHON=1
BUILD_CARGO=1
BUILD_PYDF=0
BUILD_PYDF_DATA=0
PYDF_DATA_FEATURES=""
CARGO_FLAGS="${CARGO_FLAGS:---workspace --release --all-features}"
USE_ALL=0

usage() {
  cat <<'EOF'
Usage: ./setup.sh [options]

Python (default on):
  --extras LIST             Comma-separated extras to add (dev,train,eval). Default always includes asr-mlx.
  --python-bin PATH         Python interpreter to use (default: python3.10)
  --venv DIR                Virtualenv directory (default: .venv)
  --no-python               Skip Python environment setup
  --all                     Convenience: enables extras dev,train,eval and builds pyDF + pyDF-data

Cargo (default on):
  --cargo-flags "FLAGS"     Override cargo flags (default: --workspace --release --all-features)
  --no-cargo                Skip Cargo build

Maturin bindings (optional):
  --with-pydf               Build/install pyDF via maturin develop --release -m pyDF/Cargo.toml
  --with-pydf-data          Build/install pyDF-data via maturin develop --release -m pyDF-data/Cargo.toml
  --pydf-data-hdf5-static   Build pyDF-data with --features hdf5-static (implies --with-pydf-data)

General:
  -h, --help                Show this help
Environment overrides:
  PYTHON_BIN, VENV_DIR, CARGO_FLAGS can be exported instead of flags.
EOF
}

# ------------------------- arg parse ------------------------- #
while [[ $# -gt 0 ]]; do
  case "$1" in
    --extras)
      IFS=',' read -r -a USER_EXTRAS <<<"$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --all)
      USE_ALL=1
      shift 1
      ;;
    --no-python)
      BUILD_PYTHON=0
      shift 1
      ;;
    --cargo-flags)
      CARGO_FLAGS="$2"
      shift 2
      ;;
    --no-cargo)
      BUILD_CARGO=0
      shift 1
      ;;
    --with-pydf)
      BUILD_PYDF=1
      shift 1
      ;;
    --with-pydf-data)
      BUILD_PYDF_DATA=1
      shift 1
      ;;
    --pydf-data-hdf5-static)
      BUILD_PYDF_DATA=1
      PYDF_DATA_FEATURES="--features hdf5-static"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

# Apply --all bundle
if [[ $USE_ALL -eq 1 ]]; then
  USER_EXTRAS+=("dev" "train" "eval")
  BUILD_PYDF=1
  BUILD_PYDF_DATA=1
fi

# ------------------------- helpers ------------------------- #
require_cmd() {
  local cmd="$1" desc="$2"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: $desc ('$cmd') not found in PATH." >&2
    exit 1
  fi
}

dedupe_extras() {
  local -a raw=("$@")
  local -A seen=()
  local out=()
  for e in "${raw[@]}"; do
    [[ -z "$e" ]] && continue
    if [[ -z "${seen[$e]:-}" ]]; then
      seen[$e]=1
      out+=("$e")
    fi
  done
  printf "%s\n" "${out[@]}"
}

# ------------------------- Python ------------------------- #
if [[ $BUILD_PYTHON -eq 1 ]]; then
  echo "==> Python setup (venv: $VENV_DIR; python: $PYTHON_BIN)"
  require_cmd "$PYTHON_BIN" "Python 3.10+"

  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  python -m pip install -U pip setuptools wheel

  all_extras=("${DEFAULT_EXTRAS[@]}")
  if [[ ${#USER_EXTRAS[@]} -gt 0 ]]; then
    all_extras+=("${USER_EXTRAS[@]}")
  fi
  mapfile -t uniq_extras < <(dedupe_extras "${all_extras[@]}")

  extras_str=""
  if [[ ${#uniq_extras[@]} -gt 0 ]]; then
    IFS=',' read -r extras_str <<<"$(printf "%s," "${uniq_extras[@]}" | sed 's/,$//')"
  fi

  if [[ -n "$extras_str" ]]; then
    echo "Installing project with extras: [$extras_str]"
    python -m pip install ".[${extras_str}]"
  else
    echo "Installing project without extras"
    python -m pip install .
  fi
fi

# ------------------------- Cargo ------------------------- #
if [[ $BUILD_CARGO -eq 1 ]]; then
  echo "==> Cargo build ($CARGO_FLAGS)"
  require_cmd cargo "Cargo"
  require_cmd rustc "rustc"
  cargo --version
  rustc --version
  cargo build $CARGO_FLAGS
fi

# ------------------------- Maturin bindings ------------------------- #
if [[ $BUILD_PYDF -eq 1 || $BUILD_PYDF_DATA -eq 1 ]]; then
  echo "==> Maturin builds"
  if ! command -v maturin >/dev/null 2>&1; then
    echo "maturin not found; installing into current environment"
    python -m pip install maturin
  fi
fi

if [[ $BUILD_PYDF -eq 1 ]]; then
  echo "  - Building pyDF (maturin develop --release -m pyDF/Cargo.toml)"
  maturin develop --release -m pyDF/Cargo.toml
fi

if [[ $BUILD_PYDF_DATA -eq 1 ]]; then
  echo "  - Building pyDF-data (maturin develop --release $PYDF_DATA_FEATURES -m pyDF-data/Cargo.toml)"
  maturin develop --release $PYDF_DATA_FEATURES -m pyDF-data/Cargo.toml
fi

echo "==> Done."
if [[ $BUILD_PYTHON -eq 1 ]]; then
  echo "To activate later: source \"$VENV_DIR/bin/activate\""
fi
