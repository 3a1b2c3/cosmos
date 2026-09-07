#!/bin/bash
# Build the venv for make_seg_control.py: SAM 2 plus a CUDA torch.
#
# The Linux counterpart of setup.bat. It uses uv's --torch-backend rather than
# an explicit index, since that resolves the right aarch64 wheels on the GB300;
# the vocabulary stops at cu130, which runs fine on a 13.x driver because CUDA
# minor versions are forward compatible.
#
#   bash setup.sh                          # cu130 wheels
#   TORCH_BACKEND=cu128 bash setup.sh      # a different CUDA build
#   PYTHON=3.12 bash setup.sh              # a different interpreter
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/.venv}"
PYTHON="${PYTHON:-3.11}"
TORCH_BACKEND="${TORCH_BACKEND:-cu130}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

echo "=========================================="
echo "segmap setup"
echo "  venv:   $VENV"
echo "  python: $PYTHON"
echo "  torch:  --torch-backend=$TORCH_BACKEND"
echo "=========================================="
echo

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not on PATH. Install it from https://astral.sh/uv" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Video propagation on CPU is impractically slow." >&2
  exit 1
fi

echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv
echo

echo "[1/4] Creating the venv on Python $PYTHON..."
uv venv "$VENV" --python "$PYTHON" --seed --managed-python --allow-existing
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[2/4] Installing torch..."
# Installed first and on its own: a later resolve that pulls torch as a
# transitive dependency can otherwise replace this with a CPU wheel, which
# surfaces much later as "Torch not compiled with CUDA enabled".
uv pip install --torch-backend="$TORCH_BACKEND" torch torchvision

echo "[3/4] Installing SAM 2 and the media dependencies..."
# SAM2_BUILD_CUDA=0 skips the optional mask-postprocessing extension, which
# needs nvcc to build and is not required for video propagation.
# Bare git URL, not "sam2 @ git+...": upstream declares the distribution as
# SAM-2 while the import package is sam2, and a named direct reference is
# rejected for the mismatch.
# huggingface_hub is listed explicitly: sam2 imports it inside from_pretrained
# but does not declare it as a dependency, so the failure only appears at the
# first checkpoint download rather than at install time.
# transformers carries SegFormer, used by make_seg_control_semantic.py for the
# Cityscapes-class control videos. SAM 2 does not need it.
SAM2_BUILD_CUDA=0 uv pip install "git+https://github.com/facebookresearch/sam2.git" huggingface_hub hydra-core imageio imageio-ffmpeg iopath numpy opencv-python-headless transformers

echo "[4/4] Verifying..."
python -c "import torch, sam2; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} built_for={torch.version.cuda}'); print('sam2 ok'); raise SystemExit(0 if torch.cuda.is_available() else 'torch cannot see the GPU: it resolved to a CPU build, or the driver predates it')"

echo
echo "=========================================="
echo "Done. Run:"
echo "  $VENV/bin/python $HERE/make_seg_control.py <video> -o control_seg.mp4"
echo "=========================================="
