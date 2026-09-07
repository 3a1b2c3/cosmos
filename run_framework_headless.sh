#!/bin/bash
# Run the Cosmos 3 video-transfer notebook that uses the *Cosmos Framework*
# (not Diffusers) on a headless box.
#
# This is a different path from run_transfer_headless.sh and shares nothing
# with it: the notebook clones NVIDIA/cosmos-framework and builds its own venv,
# then every inference cell shells out to that venv. Do not run one expecting
# the other's environment.
#
#   bash run_framework_headless.sh                 # Nano only
#   MODELS=all bash run_framework_headless.sh      # Nano and Super (multi-GPU)
#   KEEP_APT=1 bash run_framework_headless.sh      # keep the apt-get cell (root)
#   DRIVER_ONLY=1 bash run_framework_headless.sh   # build the driver env and stop
#
# Outputs land in cookbooks/cosmos3/generator/transfer/outputs/notebooks/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFER_DIR="$HERE/cookbooks/cosmos3/generator/transfer"
NOTEBOOK="$TRANSFER_DIR/run_video_transfer_with_cosmos_framework.ipynb"
# Only drives the notebook. The real work happens in the framework's own venv,
# built by the notebook's §6 cell, so this stays deliberately small.
DRIVER_VENV="${DRIVER_VENV:-$HERE/.venv-notebook-driver}"
KERNEL_NAME="cosmos3-driver"
# The notebook runs Nano, then Super, then a multi-control Nano section. Super
# is 32B and multi-GPU, so it is dropped unless asked for.
MODELS="${MODELS:-nano}"

# The notebook's §2 cell points HF_HOME at a cache under the cookbook directory.
# Left alone that re-downloads Cosmos3-Nano and the Guardrail even when both are
# already in the user cache, so the default is overridden to the usual location.
export COSMOS3_HF_HOME="${COSMOS3_HF_HOME:-$HOME/.cache/huggingface}"
# The HF cache serves files through symlinks, which the host's TOCTOU guard
# refuses to follow at open time. Real files instead. See RUNNING.md.
export HF_HUB_DISABLE_SYMLINKS="${HF_HUB_DISABLE_SYMLINKS:-1}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

echo "=========================================="
echo "Cosmos 3 video transfer, framework path, headless"
echo "  notebook:  ${NOTEBOOK#"$HERE/"}"
echo "  driver:    $DRIVER_VENV"
echo "  models:    $MODELS"
echo "  hf cache:  $COSMOS3_HF_HOME"
echo "=========================================="
echo

if [ ! -f "$NOTEBOOK" ]; then
  echo "ERROR: $NOTEBOOK is missing. Run this from a cosmos checkout." >&2
  exit 1
fi
for tool in uv git nvidia-smi; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: $tool is not on PATH." >&2
    exit 1
  fi
done
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set. The framework downloads gated weights." >&2
  echo "  export HF_TOKEN=<token>       (do not paste it into the notebook)" >&2
  exit 1
fi

echo "--- GPU ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo

echo "[1/3] Building the driver environment..."
uv venv "$DRIVER_VENV" --python 3.12 --seed --managed-python --allow-existing
# shellcheck disable=SC1091
source "$DRIVER_VENV/bin/activate"
uv pip install ipykernel jupyter papermill
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Cosmos3 notebook driver"

if [ "${DRIVER_ONLY:-0}" = "1" ]; then
  echo
  echo "Driver ready (DRIVER_ONLY=1). Run again without it to execute."
  exit 0
fi

echo "[2/3] Preparing the notebook..."
# Two cells are dropped even in "all" mode, and the Super sections in "nano":
#
#   §4  apt-get install, which runs without sudo and aborts the cell under
#       set -e for any non-root user. Kept with KEEP_APT=1, or when already
#       root, since in a container it is the correct thing to run.
#   §6.5 huggingface-cli login --token, which writes the token to disk. HF_TOKEN
#       is already in the environment and the framework reads it from there.
RUN_NOTEBOOK="${NOTEBOOK%.ipynb}.headless.ipynb"
KEEP_APT="${KEEP_APT:-0}"
if [ "$(id -u)" = "0" ]; then
  KEEP_APT=1
fi
KEEP_APT="$KEEP_APT" MODELS="$MODELS" python - "$NOTEBOOK" "$RUN_NOTEBOOK" <<'FILTER'
import json
import os
import re
import sys

source_path, target_path = sys.argv[1], sys.argv[2]
keep_apt = os.environ["KEEP_APT"] == "1"
nano_only = os.environ["MODELS"] == "nano"

with open(source_path, encoding="utf-8") as handle:
    notebook = json.load(handle)

# A cell belongs to the most recent "## " heading. Super sections are headed
# "## Super Inference" and "## 14. Super: ...", while the multi-control section
# that follows them is Nano and headed "## 19. ...", so tracking the heading is
# what keeps that section when Super is dropped. Truncating at the first Super
# heading would lose it.
super_heading = re.compile(r"^##\s+(Super Inference|\d+\.\s*Super:)")
kept, dropped = [], []
in_super = False

for cell in notebook["cells"]:
    source = "".join(cell["source"])
    if cell["cell_type"] == "markdown":
        for line in source.splitlines():
            if line.startswith("## "):
                in_super = bool(super_heading.match(line))
                break
    if nano_only and in_super:
        dropped.append("super")
        continue
    if cell["cell_type"] == "code":
        if not keep_apt and "apt-get install" in source:
            dropped.append("apt")
            continue
        if "login --token" in source:
            dropped.append("hf-login")
            continue
    kept.append(cell)

notebook["cells"] = kept
with open(target_path, "w", encoding="utf-8") as handle:
    json.dump(notebook, handle, indent=1)

summary = ", ".join(f"{reason} x{dropped.count(reason)}" for reason in sorted(set(dropped))) or "none"
print(f"  {len(kept)} cells kept, dropped: {summary}")
FILTER
if [ "$KEEP_APT" != "1" ]; then
  echo "      The apt-get cell was dropped. If a cell fails on libxcb/libGL, run:"
  echo "        sudo apt-get install -y curl ffmpeg git-lfs libgl1 libglib2.0-0 libx11-dev libxcb1 wget"
fi

echo "[3/3] Executing the notebook..."
# papermill streams each cell's output as it is produced; nbconvert buffers
# until a cell returns, which makes a long generation step indistinguishable
# from a hang. Run from the notebook's own directory: several cells resolve
# preview_helpers and the spec files relative to the working directory.
EXECUTED="${RUN_NOTEBOOK%.ipynb}.executed.ipynb"
cd "$TRANSFER_DIR"
papermill "$RUN_NOTEBOOK" "$EXECUTED" --kernel "$KERNEL_NAME" --log-output --log-level INFO

echo
echo "=========================================="
echo "Done. Generated clips:"
echo "=========================================="
CLIPS="$(find "$TRANSFER_DIR/outputs" -name "*.mp4" 2>/dev/null | sort)"
if [ -n "$CLIPS" ]; then
  echo "$CLIPS" | sed "s|^|  |"
else
  echo "  none found under ${TRANSFER_DIR#"$HERE/"}/outputs/"
fi
