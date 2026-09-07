#!/bin/bash
# Run the Cosmos 3 video-transfer (video2video) notebook on a headless box.
#
# The notebook installs its own venv and then asks you to switch the Jupyter
# kernel by hand, which needs a browser. This does the install as a shell step
# and executes the notebook with that kernel selected, so nothing is
# interactive.
#
#   bash run_transfer_headless.sh                  # install, then run Nano
#   MODELS=all bash run_transfer_headless.sh       # Nano and Super
#   SKIP_INSTALL=1 bash run_transfer_headless.sh   # env already built
#   INSTALL_ONLY=1 bash run_transfer_headless.sh   # build the env and stop
#   TORCH_BACKEND=cu128 bash run_transfer_headless.sh
#
# Outputs land in outputs/notebooks/diffusers/<model>/<spec>/vision.mp4
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOTEBOOK="$HERE/cookbooks/cosmos3/generator/transfer/run_video_transfer_with_diffusers.ipynb"
# The notebook's own default, kept so both paths share one environment.
COSMOS3_DIFFUSERS_VENV="${COSMOS3_DIFFUSERS_VENV:-$HERE/.venv-cosmos3-diffusers}"
# uv's --torch-backend vocabulary ends at cu130 and rejects cu132 outright. A
# 13.x driver runs cu130 wheels, since CUDA minor versions are forward
# compatible. Use setup_diffusers.sh instead if you need the cu132 index.
TORCH_BACKEND="${TORCH_BACKEND:-cu130}"
KERNEL_NAME="cosmos3-diffusers"
# The notebook runs Nano and then Super. Super is 32B and multi-GPU, so it
# is skipped unless asked for: on one GPU those cells fail after the Nano
# work has already succeeded.
MODELS="${MODELS:-nano}"
export COSMOS3_DIFFUSERS_VENV COSMOS3_TORCH_BACKEND="$TORCH_BACKEND"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

echo "=========================================="
echo "Cosmos 3 video transfer, headless"
echo "  notebook: ${NOTEBOOK#"$HERE/"}"
echo "  venv:     $COSMOS3_DIFFUSERS_VENV"
echo "  torch:    --torch-backend=$TORCH_BACKEND"
echo "  models:   $MODELS"
echo "  hf cache: ${HF_HOME:-<default, set HF_HOME for a roomier disk>}"
echo "=========================================="
echo

if [ ! -f "$NOTEBOOK" ]; then
  echo "ERROR: $NOTEBOOK is missing. Run this from a cosmos checkout." >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not on PATH. Install it from https://astral.sh/uv" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; this needs an NVIDIA GPU." >&2
  exit 1
fi
# A missing libxcb surfaces as an import error that looks nothing like a
# graphics problem, and headless images routinely lack it.
if ! ldconfig -p 2>/dev/null | grep -q "libxcb.so.1"; then
  echo "WARNING: libxcb.so.1 was not found. If a cell fails on import, run:" >&2
  echo "  sudo apt-get install -y libxcb1 libgl1 libglib2.0-0" >&2
  echo
fi

echo "--- GPU ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo

if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  echo "[1/3] Building the environment and registering the '$KERNEL_NAME' kernel..."
  uv venv "$COSMOS3_DIFFUSERS_VENV" --python 3.13 --seed --managed-python --allow-existing
  # shellcheck disable=SC1091
  source "$COSMOS3_DIFFUSERS_VENV/bin/activate"
  uv pip install --torch-backend="$TORCH_BACKEND" "diffusers @ git+https://github.com/huggingface/diffusers.git" accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg ipykernel jupyter papermill torch torchvision transformers
  python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Cosmos3 Diffusers (Python 3.13)"
else
  echo "[1/3] Reusing the existing environment (SKIP_INSTALL=1)."
  # shellcheck disable=SC1091
  source "$COSMOS3_DIFFUSERS_VENV/bin/activate"
fi

echo "[2/3] Verifying the environment..."
python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} built_for={torch.version.cuda}'); raise SystemExit(0 if torch.cuda.is_available() else 'torch cannot see the GPU: it resolved to a CPU build, or the driver predates it')"
if ! python -c "import huggingface_hub, sys; sys.exit(0 if huggingface_hub.get_token() else 1)" 2>/dev/null; then
  echo "ERROR: no Hugging Face token, so the checkpoint download will fail." >&2
  echo "  uvx hf@latest auth login   (or export HF_TOKEN)" >&2
  exit 1
fi
# Edge and depth need no gated access; blur, seg and WSM use the Guardrail.
echo "      Blur, segmentation and world-scenario-map need access to the gated"
echo "      nvidia/Cosmos-1.0-Guardrail repository. Edge and depth do not."

if [ "${INSTALL_ONLY:-0}" = "1" ]; then
  echo
  echo "Environment ready (INSTALL_ONLY=1). Run again without it to execute."
  exit 0
fi

RUN_NOTEBOOK="$NOTEBOOK"
if [ "$MODELS" = "nano" ]; then
  # Executed as a filtered copy so the checked-in notebook keeps its Super
  # cells and stays unmodified by --inplace.
  RUN_NOTEBOOK="${NOTEBOOK%.ipynb}.nano.ipynb"
  python - "$NOTEBOOK" "$RUN_NOTEBOOK" <<'FILTER'
import json
import sys

source_path, target_path = sys.argv[1], sys.argv[2]
with open(source_path, encoding="utf-8") as handle:
    notebook = json.load(handle)

# Cells from the first "## Super:" heading onward belong to the 32B model.
cells = notebook["cells"]
first_super = next(
    (
        index
        for index, cell in enumerate(cells)
        if cell["cell_type"] == "markdown" and "".join(cell["source"]).lstrip().startswith("## Super:")
    ),
    len(cells),
)
cells = cells[:first_super]

# The notebook builds its own environment in a %%bash cell. This script has
# already done that, and re-resolving the same packages costs minutes of
# silence on every run, so the cell is dropped rather than repeated.
kept = []
for cell in cells:
    source = "".join(cell["source"])
    if cell["cell_type"] == "code" and "uv pip install" in source and "ipykernel install" in source:
        continue
    kept.append(cell)
dropped = len(cells) - len(kept)
notebook["cells"] = kept
with open(target_path, "w", encoding="utf-8") as handle:
    json.dump(notebook, handle, indent=1)
print(f"  Nano-only copy: {len(kept)} cells, {dropped} install cell(s) dropped")
FILTER
fi

echo "[3/3] Executing the notebook..."
# The kernel has to be named explicitly either way: the notebook's own
# kernelspec is plain python3, so without it every cell runs outside the venv
# and the notebook's guard cell raises.
#
# papermill is preferred because it streams each cell's output as it is
# produced. nbconvert buffers until a cell returns, so a diffusion step that
# takes minutes prints nothing and cannot be told apart from a hang.
EXECUTED="${RUN_NOTEBOOK%.ipynb}.executed.ipynb"
if command -v papermill >/dev/null 2>&1; then
  papermill "$RUN_NOTEBOOK" "$EXECUTED" --kernel "$KERNEL_NAME" --log-output --log-level INFO
else
  echo "  papermill not found; falling back to nbconvert (no live output)."
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name="$KERNEL_NAME" --ExecutePreprocessor.timeout=-1 "$RUN_NOTEBOOK"
fi

echo
echo "=========================================="
echo "Done. Generated clips:"
echo "=========================================="
find "$HERE/outputs/notebooks/diffusers" -name "vision.mp4" 2>/dev/null | sed "s|^|  |" || echo "  (none found under outputs/notebooks/diffusers)"
