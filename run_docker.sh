#!/bin/bash
# Open a shell in the NGC PyTorch container the Cosmos README recommends, with
# this repository and the Hugging Face cache mounted.
#
# On a DGX this beats a venv: the image ships a torch already built for the
# right CUDA and the right architecture, so none of the wheel matching that
# setup_diffusers.sh has to guard against applies. It is also the only way to
# reach the vLLM-Omni, SGLang and TensorRT-LLM paths.
#
#   bash run_docker.sh                        # shell in the CUDA 13 image
#   bash run_docker.sh python my_script.py    # run something and exit
#   IMAGE=nvcr.io/nvidia/pytorch:25.06-py3 bash run_docker.sh   # CUDA 12
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 25.09 is the CUDA 13 image, 25.06 the CUDA 12 one. The container's CUDA major
# version has to match the driver's.
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.09-py3}"
# Kept outside the container so checkpoints survive it, and so a second run
# does not download Cosmos again.
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not on PATH." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: the docker daemon is not reachable. Is it running, and are you" >&2
  echo "       in the docker group?" >&2
  exit 1
fi
# Without the container toolkit the run starts but has no GPU, which surfaces
# later as torch.cuda.is_available() being False rather than as a docker error.
if ! docker run --rm --gpus all "$IMAGE" nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: the container cannot see a GPU. Install the NVIDIA Container" >&2
  echo "       Toolkit, or check that '$IMAGE' is pulled and you are logged" >&2
  echo "       in to nvcr.io." >&2
  exit 1
fi

mkdir -p "$HF_CACHE"
echo "=========================================="
echo "image:     $IMAGE"
echo "repo:      $HERE -> /workspace/cosmos"
echo "hf cache:  $HF_CACHE -> /root/.cache/huggingface"
echo "=========================================="
echo

# --ipc=host: the default 64MB /dev/shm is too small for dataloader workers.
# HF_TOKEN is passed through rather than baked in, so it stays out of the image
# and out of shell history.
exec docker run --rm -it --gpus all --ipc=host --network host --volume "$HERE:/workspace/cosmos" --volume "$HF_CACHE:/root/.cache/huggingface" --env "HF_TOKEN=${HF_TOKEN:-}" --workdir /workspace/cosmos "$IMAGE" "$@"
