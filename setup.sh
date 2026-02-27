#!/usr/bin/env bash
# One-step builder for DeepFilterNet4 (Python + Cargo + optional maturin bindings).
# Defaults:
#   - Python: create .venv with python3.10 and install .[asr-mlx]
#   - Cargo:  cargo build --workspace --release --all-features
# Use flags below to toggle extras or skip parts.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/scripts/install-hooks.sh" ]]; then
  "$ROOT_DIR/scripts/install-hooks.sh"
fi

# ------------------------- platform detection ------------------------- #
PLATFORM_OS="$(uname -s)"
PLATFORM_ARCH="$(uname -m)"

is_darwin_arm64() {
  [[ "$PLATFORM_OS" == "Darwin" && "$PLATFORM_ARCH" == "arm64" ]]
}

warn_mlx_platform() {
  if ! is_darwin_arm64; then
    cat >&2 <<'MLXWARN'
╔═══════════════════════════════════════════════════════════════════════════╗
║ WARNING: MLX packages require macOS on Apple Silicon (Darwin arm64).      ║
║ Current platform: OS=$PLATFORM_OS, ARCH=$PLATFORM_ARCH                    ║
║                                                                           ║
║ The 'asr-mlx' extra will only install 'openai-whisper'; mlx, mlx-whisper, ║
║ and mlx-data will be silently skipped due to platform markers.            ║
║                                                                           ║
║ Consider using --extras "asr" instead for cross-platform ASR support.     ║
╚═══════════════════════════════════════════════════════════════════════════╝
MLXWARN
    cat >&2 <<EOF
(Detected: OS=$PLATFORM_OS, ARCH=$PLATFORM_ARCH)
EOF
  fi
}

# ------------------------- defaults ------------------------- #
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
DEFAULT_EXTRAS=("asr-mlx" "train" "eval")
USER_EXTRAS=()
BUILD_PYTHON=1
BUILD_CARGO=1
BUILD_PYDF=0
BUILD_PYDF_DATA=0
BUILD_PYDF_AUGMENT=0
PYDF_DATA_FEATURES=""
CARGO_FLAGS="${CARGO_FLAGS:---workspace --release --all-features}"
USE_ALL=0
LD_OVERRIDE=""
CARGO_INCLUDE_PYDF=0
DRY_RUN=0
PRINT_ENV=0
PY_EDITABLE="${PY_EDITABLE:-0}"
# Normalize env-supplied PY_EDITABLE to 0 or 1 before any numeric comparisons.
case "${PY_EDITABLE}" in
  1|true|yes|on)  PY_EDITABLE=1 ;;
  0|false|no|off) PY_EDITABLE=0 ;;
  *)
    echo "ERROR: PY_EDITABLE must be 0 or 1 (got '${PY_EDITABLE}')" >&2
    exit 2
    ;;
esac
PY_EDITABLE_EXPLICIT=0

usage() {
  cat <<'EOF'
Usage: ./setup.sh [options]

Python (default on):
  --extras LIST             Comma-separated extras to add (dev,train,eval). Default always includes asr-mlx.
  --python-bin PATH         Python interpreter to use (default: python3.10)
  --venv DIR                Virtualenv directory (default: .venv)
  --no-python               Skip Python environment setup
  --(no-)editable           Install Python packages in editable mode (pip install -e .)
  --all                     Convenience: enables extras dev,train,eval and builds pyDF + pyDF-data + pyDF-augment; also sets editable mode for Python unless --no-editable is given.

Cargo (default on):
  --cargo-flags "FLAGS"     Override cargo flags (default: --workspace --release --all-features)
  --no-cargo                Skip Cargo build
  --ld PATH                 Override ld linker path (exported as LD)
  --cargo-include-pydf      Include pyDF/pyDF-data crates in cargo build (default: excluded)

Maturin bindings (optional):
  --with-pydf               Build/install pyDF via maturin develop --release -m pyDF/Cargo.toml
  --with-pydf-data          Build/install pyDF-data via maturin develop --release -m pyDF-data/Cargo.toml
  --pydf-data-hdf5-static   Build pyDF-data with --features hdf5-static (implies --with-pydf-data)
  --with-pydf-augment       Build/install pyDF-augment via maturin develop --release -m pyDF-augment/Cargo.toml

Diagnostic:
  --print-env               Print detected environment and exit (no build)
  --dry-run                 Show what would be done without executing

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
    --ld)
      LD_OVERRIDE="$2"
      shift 2
      ;;
    --cargo-include-pydf)
      CARGO_INCLUDE_PYDF=1
      shift 1
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
    --with-pydf-augment)
      BUILD_PYDF_AUGMENT=1
      shift 1
      ;;
    --print-env)
      PRINT_ENV=1
      shift 1
      ;;
    --dry-run)
      DRY_RUN=1
      shift 1
      ;;
    --editable)
      PY_EDITABLE=1
      PY_EDITABLE_EXPLICIT=1
      shift 1
      ;;
    --no-editable)
      PY_EDITABLE=0
      PY_EDITABLE_EXPLICIT=1
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
  BUILD_PYDF_AUGMENT=1
  if [[ $PY_EDITABLE_EXPLICIT -eq 0 ]]; then
    PY_EDITABLE=1
  fi
fi

# ------------------------- print-env ------------------------- #
if [[ $PRINT_ENV -eq 1 ]]; then
  echo "=== DeepFilterNet setup.sh Environment Report ==="
  echo "Platform:"
  echo "  OS:       $PLATFORM_OS"
  echo "  ARCH:     $PLATFORM_ARCH"
  echo "  MLX OK:   $(is_darwin_arm64 && echo "YES" || echo "NO (MLX packages will be skipped)")"
  echo ""
  echo "Python:"
  echo "  PYTHON_BIN: $PYTHON_BIN"
  echo "  VENV_DIR:   $VENV_DIR"
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "  Version:    $("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
    echo "  Path:       $(command -v "$PYTHON_BIN")"
  else
    echo "  ERROR:      '$PYTHON_BIN' not found!"
  fi
  echo ""
  echo "Cargo:"
  echo "  FLAGS: $CARGO_FLAGS"
  if command -v cargo >/dev/null 2>&1; then
    echo "  cargo: $(cargo --version)"
    echo "  rustc: $(rustc --version)"
  else
    echo "  ERROR: cargo not found!"
  fi
  echo ""
  echo "Extras:"
  echo "  DEFAULT: ${DEFAULT_EXTRAS[*]}"
  echo "  USER:    ${USER_EXTRAS[*]:-<none>}"
  echo ""
  echo "Build Flags:"
  echo "  BUILD_PYTHON:    $BUILD_PYTHON"
  echo "  BUILD_CARGO:     $BUILD_CARGO"
  echo "  BUILD_PYDF:      $BUILD_PYDF"
  echo "  BUILD_PYDF_DATA: $BUILD_PYDF_DATA"
  echo "  BUILD_PYDF_AUG:  $BUILD_PYDF_AUGMENT"
  echo "  PY_EDITABLE:     $PY_EDITABLE"
  exit 0
fi

# ------------------------- dry-run preview ------------------------- #
if [[ $DRY_RUN -eq 1 ]]; then
  echo "=== DRY RUN MODE - Showing what would be executed ==="
  echo ""
  echo "Platform: $PLATFORM_OS / $PLATFORM_ARCH"
  echo ""

  if [[ $BUILD_PYTHON -eq 1 ]]; then
    echo "[WOULD] Create/use venv at: $VENV_DIR"
    edit_msg=""; [[ $PY_EDITABLE -eq 1 ]] && edit_msg=" (editable)"
    echo "[WOULD] Install extras: ${DEFAULT_EXTRAS[*]} ${USER_EXTRAS[*]:-}${edit_msg}"

    # Warn about MLX if applicable
    for extra in "${DEFAULT_EXTRAS[@]}" "${USER_EXTRAS[@]}"; do
      if [[ "$extra" == "mlx" || "$extra" == "asr-mlx" ]]; then
        if ! is_darwin_arm64; then
          echo "[WARN]  MLX packages will be SKIPPED (requires Darwin arm64)"
        fi
        break
      fi
    done
  else
    echo "[SKIP]  Python setup (--no-python)"
  fi
  echo ""

  if [[ $BUILD_CARGO -eq 1 ]]; then
    echo "[WOULD] cargo build $CARGO_FLAGS"
    if [[ $CARGO_INCLUDE_PYDF -eq 0 ]]; then
      echo "        (excluding DeepFilterLib, DeepFilterDataLoader, DeepFilterAugment)"
    fi
  else
    echo "[SKIP]  Cargo build (--no-cargo)"
  fi
  echo ""

  if [[ $BUILD_PYDF -eq 1 ]]; then
    echo "[WOULD] maturin develop --release -m pyDF/Cargo.toml"
  fi
  if [[ $BUILD_PYDF_DATA -eq 1 ]]; then
    echo "[WOULD] maturin develop --release ${PYDF_DATA_FEATURES:-} -m pyDF-data/Cargo.toml"
  fi
  if [[ $BUILD_PYDF_AUGMENT -eq 1 ]]; then
    echo "[WOULD] maturin develop --release -m pyDF-augment/Cargo.toml"
  fi

  echo ""
  echo "=== END DRY RUN ==="
  exit 0
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

resolve_ld() {
  if [[ -n "$LD_OVERRIDE" ]]; then
    echo "$LD_OVERRIDE"
    return
  fi
  if [[ "$(uname)" == "Darwin" ]] && command -v xcrun >/dev/null 2>&1; then
    local mac_ld
    mac_ld="$(xcrun -f ld 2>/dev/null || true)"
    if [[ -n "$mac_ld" && -x "$mac_ld" ]]; then
      echo "$mac_ld"
      return
    fi
  fi
  if command -v ld >/dev/null 2>&1; then
    command -v ld
    return
  fi
  echo ""
}

uv_avail() {
  command -v uv >/dev/null 2>&1
}

pip_install() {
  if uv_avail; then
    # Tie uv installs to the active interpreter/venv for pip parity.
    uv pip install --python "$(command -v python)" "$@"
  else
    python -m pip install "$@"
  fi
}

# ------------------------- macOS deployment target ------------------------- #
# Set early so it applies to all native builds (pip, cargo, maturin).
# This avoids version mismatch errors where assembly/C code compiled for
# the host macOS version (e.g. 26.2) fails to link against Rust's default
# target (e.g. 11.0). Affects: tract-linalg, hdf5-src, and others.
if [[ "$PLATFORM_OS" == "Darwin" ]]; then
  if [[ -z "${MACOSX_DEPLOYMENT_TARGET:-}" ]]; then
    export MACOSX_DEPLOYMENT_TARGET="14.0"
    echo "Setting MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET (native code compatibility)"
  fi
fi

# ------------------------- Python ------------------------- #
if [[ $BUILD_PYTHON -eq 1 ]]; then
  echo "==> Python setup (venv: $VENV_DIR; python: $PYTHON_BIN)"
  require_cmd "$PYTHON_BIN" "Python 3.10+"
  py_ver="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
  if [[ "$py_ver" != "3.10" ]]; then
    echo "ERROR: Python 3.10 required. '$PYTHON_BIN' reports $py_ver." >&2
    exit 1
  fi

  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  venv_ver="$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
  if [[ "$venv_ver" != "3.10" ]]; then
    echo "ERROR: venv python version $venv_ver (expected 3.10). Recreate $VENV_DIR with python3.10." >&2
    exit 1
  fi

  pushd DeepFilterNet

  pip_install -U pip setuptools wheel silero-vad

  all_extras=("${DEFAULT_EXTRAS[@]}")
  if [[ ${#USER_EXTRAS[@]} -gt 0 ]]; then
    all_extras+=("${USER_EXTRAS[@]}")
  fi
  mapfile -t uniq_extras < <(dedupe_extras "${all_extras[@]}")

  # Warn if mlx-related extras are used on non-ARM64 Darwin
  for extra in "${uniq_extras[@]}"; do
    if [[ "$extra" == "mlx" || "$extra" == "asr-mlx" ]]; then
      warn_mlx_platform
      break
    fi
  done

  extras_str=""
  if [[ ${#uniq_extras[@]} -gt 0 ]]; then
    IFS=',' read -r extras_str <<<"$(printf "%s," "${uniq_extras[@]}" | sed 's/,$//')"
  fi

  pip_install_args=()
  if [[ $PY_EDITABLE -eq 1 ]]; then
    pip_install_args+=(-e)
  fi

  if [[ -n "$extras_str" ]]; then
    edit_label=""; [[ $PY_EDITABLE -eq 1 ]] && edit_label=" (editable)"
    echo "Installing project with extras: [$extras_str]${edit_label}"
    pip_install_args+=(".[${extras_str}]")
  else
    edit_label=""; [[ $PY_EDITABLE -eq 1 ]] && edit_label=" (editable)"
    echo "Installing project without extras${edit_label}"
    pip_install_args+=(".")
  fi
  pip_install "${pip_install_args[@]}"

  popd
fi

# ------------------------- Cargo ------------------------- #
if [[ $BUILD_CARGO -eq 1 ]]; then
  build_flags=()
  read -r -a build_flags <<<"$CARGO_FLAGS"
  # Always exclude PyO3 extension modules from standalone cargo build.
  # They MUST be built via maturin which handles Python symbol resolution correctly.
  # The --cargo-include-pydf flag is deprecated and ignored for this reason.
  build_flags+=(--exclude DeepFilterLib --exclude DeepFilterDataLoader --exclude DeepFilterAugment)
  if [[ $CARGO_INCLUDE_PYDF -eq 1 ]]; then
    echo "NOTE: --cargo-include-pydf is deprecated. PyO3 extensions are built via maturin only."
  fi

  echo "==> Cargo build (${build_flags[*]})"
  require_cmd cargo "Cargo"
  require_cmd rustc "rustc"

  LD_PATH="$(resolve_ld)"
  if [[ -n "$LD_PATH" ]]; then
    export LD="$LD_PATH"
    echo "Using linker: $LD"
  else
    echo "WARNING: ld not found; build may fail if linker is missing."
  fi
  cargo --version
  rustc --version
  cargo build "${build_flags[@]}"
fi

# ------------------------- Maturin bindings ------------------------- #
if [[ $BUILD_PYDF -eq 1 || $BUILD_PYDF_DATA -eq 1 || $BUILD_PYDF_AUGMENT -eq 1 ]]; then
  echo "==> Maturin builds"
  if ! command -v maturin >/dev/null 2>&1; then
    echo "maturin not found; installing into current environment"
    pip_install maturin
  fi
fi

if [[ $BUILD_PYDF -eq 1 ]]; then
  echo "  - Building pyDF (maturin develop --release -m pyDF/Cargo.toml)"
  maturin develop --release -m pyDF/Cargo.toml
fi

if [[ $BUILD_PYDF_DATA -eq 1 ]]; then
  # hdf5-rust rev pinned by this repo supports HDF5 1.x.
  # Homebrew now defaults to hdf5 2.x, which causes pyDF-data dynamic linking to fail.
  # Detect this case and switch to static HDF5 build proactively.
  if [[ "$PLATFORM_OS" == "Darwin" && "$PYDF_DATA_FEATURES" != *"hdf5-static"* ]] \
    && command -v brew >/dev/null 2>&1; then
    brew_hdf5_prefix="$(brew --prefix hdf5 2>/dev/null || true)"
    brew_hdf5_cfg="${brew_hdf5_prefix}/include/H5pubconf.h"
    if [[ -f "$brew_hdf5_cfg" ]]; then
      brew_hdf5_version="$(awk -F'"' '/#define H5_VERSION / { print $2; exit }' "$brew_hdf5_cfg")"
      if [[ "$brew_hdf5_version" == 2.* ]]; then
        echo "Detected Homebrew HDF5 ${brew_hdf5_version}; using hdf5-static for pyDF-data."
        PYDF_DATA_FEATURES="--features hdf5-static"
      fi
    fi
  fi

  pydf_data_args=(--release -m pyDF-data/Cargo.toml)
  if [[ -n "$PYDF_DATA_FEATURES" ]]; then
    # shellcheck disable=SC2206
    pydf_data_args+=($PYDF_DATA_FEATURES)
  fi
  echo "  - Building pyDF-data (maturin develop ${pydf_data_args[*]})"
  if ! maturin develop "${pydf_data_args[@]}"; then
    if [[ "$PYDF_DATA_FEATURES" == *"hdf5-static"* ]]; then
      echo "ERROR: pyDF-data build failed even with hdf5-static enabled." >&2
      exit 1
    fi
    echo "⚠️  pyDF-data build failed with system HDF5; retrying with bundled static HDF5."
    pydf_data_retry_args=(--release --features hdf5-static -m pyDF-data/Cargo.toml)
    echo "  - Retrying pyDF-data (maturin develop ${pydf_data_retry_args[*]})"
    maturin develop "${pydf_data_retry_args[@]}"
  fi
fi

if [[ $BUILD_PYDF_AUGMENT -eq 1 ]]; then
  echo "  - Building pyDF-augment (maturin develop --release -m pyDF-augment/Cargo.toml)"
  maturin develop --release -m pyDF-augment/Cargo.toml
fi

echo "==> Done."
if [[ $BUILD_PYTHON -eq 1 ]]; then
  echo "To activate later: source \"$VENV_DIR/bin/activate\""
fi
