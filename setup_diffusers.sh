#!/bin/bash
# Set up the Cosmos 3 Diffusers path: a venv holding diffusers, torch and the
# media dependencies the Generator pipeline needs. This repository is docs and
# cookbooks -- there is nothing here to install -- so everything comes from
# PyPI and the models from Hugging Face.
#
#   bash setup_diffusers.sh                      # cu132 wheels
#   TORCH_INDEX=cu130 bash setup_diffusers.sh    # a different CUDA index
#   PYTHON=3.11 bash setup_diffusers.sh          # a different interpreter
#   KEEP_VENV=1 bash setup_diffusers.sh          # reuse an existing .venv
#
# Then: source .venv/bin/activate
#
# For a container instead, see run_docker.sh -- on a DGX that is the better
# path, since the NGC image ships a torch already built for the right CUDA and
# architecture.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
PYTHON="${PYTHON:-3.13}"
# The wheel index, not uv's --torch-backend: that option's accepted values stop
# at cu130 and it rejects cu132 outright. Note the cu132 index is sparse --
# pinning an exact torch version that only exists elsewhere resolves to a CPU
# build instead of failing, which surfaces much later as
# "Torch not compiled with CUDA enabled". Left unpinned for that reason.
TORCH_INDEX="${TORCH_INDEX:-cu132}"
TORCH_URL="https://download.pytorch.org/whl/$TORCH_INDEX"

echo "=========================================="
echo "Cosmos 3 Diffusers setup"
echo "  repo:   $HERE"
echo "  venv:   $VENV"
echo "  python: $PYTHON"
echo "  torch:  $TORCH_URL"
echo "=========================================="
echo

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not on PATH. Install it from https://astral.sh/uv" >&2
  exit 1
fi
UV_VERSION="$(uv --version | awk '{print $2}')"
if [ "$(printf '%s\n0.11.3\n' "$UV_VERSION" | sort -V | head -1)" != "0.11.3" ]; then
  echo "ERROR: uv $UV_VERSION is too old; Cosmos requires 0.11.3 or newer." >&2
  echo "  uv self update" >&2
  exit 1
fi

echo "--- GPU ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv
else
  echo "nvidia-smi not found; a CUDA build of torch will not be usable."
fi
echo

if [ -d "$VENV" ] && [ "${KEEP_VENV:-0}" != "1" ]; then
  echo "[1/5] Removing existing .venv (set KEEP_VENV=1 to reuse)..."
  rm -rf "$VENV"
fi
if [ ! -d "$VENV" ]; then
  echo "[1/5] Creating venv on Python $PYTHON..."
  uv venv "$VENV" --python "$PYTHON" --seed --managed-python
else
  echo "[1/5] Reusing existing venv."
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[2/5] Installing torch and torchvision from $TORCH_INDEX..."
uv pip install --index-url "$TORCH_URL" torch torchvision

echo "[3/5] Installing the Diffusers Generator dependencies..."
# Diffusers comes from git because Cosmos3OmniPipeline is newer than its last
# PyPI release. No index-url here: these are ordinary PyPI packages, and
# pointing them at the torch index would fail to resolve them.
uv pip install "diffusers @ git+https://github.com/huggingface/diffusers.git" accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg transformers

echo "[4/5] Verifying the CUDA build..."
python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} built_for={torch.version.cuda}'); raise SystemExit(0 if torch.cuda.is_available() else 'torch cannot see the GPU: it resolved to a CPU build, or the driver predates it')"

echo "[5/5] Checking Hugging Face access..."
if [ -n "${HF_HOME:-}" ]; then
  echo "      HF_HOME=$HF_HOME"
else
  echo "      HF_HOME is unset, so checkpoints land in the default cache."
  echo "      Cosmos3-Nano is 64B total / 16B active -- point HF_HOME at a"
  echo "      disk with room before the first run."
fi
if python -c "import huggingface_hub, sys; sys.exit(0 if huggingface_hub.get_token() else 1)" 2>/dev/null; then
  echo "      Hugging Face token found."
else
  echo "      No Hugging Face token. Authenticate before running a pipeline:"
  echo "        uvx hf@latest auth login"
fi
echo "      The Generator needs nvidia/Cosmos-1.0-Guardrail, which is gated."
echo "      Request access, or pass enable_safety_checker=False."

echo
echo "=========================================="
echo "Done. Activate with: source $VENV/bin/activate"
echo "=========================================="
echo "The README's text-to-video example downloads Cosmos3-Nano on first run"
echo "and is compute-heavy; long step times are expected, not a hang."
