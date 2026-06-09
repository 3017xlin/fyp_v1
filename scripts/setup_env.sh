#!/usr/bin/env bash
# Minimal environment setup; see README for full instructions.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$HOME/kdvit-venv}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCH_CUDA="${TORCH_CUDA:-cu121}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch==${TORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
python -m pip install "torch_geometric>=2.5"
python -m pip install -r requirements.txt
