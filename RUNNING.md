# Running Cosmos 3 here

Notes for running this checkout on our hardware, and what the three scripts in
the repo root do. The upstream instructions are in [`README.md`](./README.md);
this covers what differs for us.

## What this repository is

Documentation, cookbooks and evaluation material. There is no `pyproject.toml`
or `setup.py` — nothing to install *from* here. The models come from Hugging
Face and the code from Diffusers, vLLM-Omni, SGLang or TensorRT-LLM. "Installing
Cosmos" means building an environment that can run those.

## Scripts

| Script | Use |
| --- | --- |
| [`setup_diffusers.sh`](./setup_diffusers.sh) | Build a venv for the Diffusers path, torch from an explicit CUDA index. |
| [`run_docker.sh`](./run_docker.sh) | Shell in the NGC container, repo and HF cache mounted. |
| [`run_transfer_headless.sh`](./run_transfer_headless.sh) | Run the video-transfer notebook with no browser. |

## Which CUDA build

`uv pip install --torch-backend=...` accepts values up to **cu130**; it rejects
`cu132` outright, and there is no `cu133`. A 13.x driver runs cu130 wheels —
CUDA minor versions are forward compatible — so cu130 is the simple answer for
both boxes.

If a cu132 build is required specifically, it has to come from
`--index-url https://download.pytorch.org/whl/cu132`, which is what
`setup_diffusers.sh` does. Note that index is sparse: pinning a torch version
that only exists elsewhere resolves to a **CPU build** rather than failing, and
that surfaces much later as `Torch not compiled with CUDA enabled`.

## Video transfer (video2video) on the DGX

Headless, so the notebook's "switch the Jupyter kernel by hand" step cannot
happen. `run_transfer_headless.sh` does the install as a shell step and executes
the notebook with that kernel already selected.

```bash
sudo apt-get install -y libxcb1 libgl1 libglib2.0-0
export HF_TOKEN=<token>
uvx hf@latest download nvidia/Cosmos3-Nano    # 1.23 GB, resumable
bash run_transfer_headless.sh
```

Run it under `tmux` or `nohup`: the job outlives no SSH session on its own, and
losing the connection kills it.

```bash
tmux new -s cosmos      # Ctrl-B D to detach, tmux attach -t cosmos to return
```

### Hugging Face access

The weights are not in this repository, so a token is required before anything
runs. `HF_TOKEN` in the environment is preferred over `hf auth login`, which
writes a token file to disk.

Pull the checkpoint separately rather than letting the notebook do it. It is
resumable, it reports progress, and it separates "the download failed" from
"generation failed" — one long command that dies tells you neither.

Cosmos3-Nano is **1.23 GB across 68 files**, not the tens of gigabytes the
model-family table implies: the `64B / 16B / 4B` figures there are three
different models in adjacent columns. Downloads land in `$HF_HOME` when set and
`~/.cache/huggingface` otherwise — the run has to use whichever the download
used, or it fetches again.

| Variable | Default | Effect |
| --- | --- | --- |
| `MODELS` | `nano` | `all` also runs Cosmos3-Super, which is 32B and multi-GPU. |
| `TORCH_BACKEND` | `cu130` | Passed to `uv pip install --torch-backend`. |
| `SKIP_INSTALL` | `0` | Reuse an environment already built. |
| `INSTALL_ONLY` | `0` | Build the environment and stop. |
| `COSMOS3_DIFFUSERS_VENV` | `./.venv-cosmos3-diffusers` | Where the venv goes. |

Nano mode runs a filtered copy of the notebook, so the checked-in one keeps its
Super cells and is not rewritten in place. The filter drops two things:

- **Cells from the first `## Super:` heading onward** — the 32B model, which
  needs more than one GPU.
- **The notebook's own install cell.** It builds a venv and installs the same
  packages the script already installed, which costs minutes of silence on
  every run and produces no output while it works.

The venv guard cell is kept, so a wrong kernel still fails loudly rather than
running against the wrong interpreter.

Clips land under the cookbook directory, not the repo root:
`cookbooks/cosmos3/generator/transfer/outputs/notebooks/diffusers/<model>/<spec>/vision.mp4`.

The notebook also runs with `guardrails: False`, so the gated Guardrail
repository may not be needed for blur, segmentation and WSM after all —
despite the README listing it as a requirement.

### Keeping the scripts in sync

The scripts are edited on Windows and run on the DGX, so copy them across after
a change:

```bash
scp run_transfer_headless.sh setup_diffusers.sh RUNNING.md <user>@<dgx>:~/cosmos/
```

A run that behaves like an older version usually means this step was missed.

### Watching it run

The script executes through `papermill`, which streams each cell's output as it
is produced. `nbconvert` — the fallback when papermill is missing — buffers
until a cell returns, so a diffusion step that takes minutes prints nothing and
cannot be told apart from a hang.

Either way, from a second pane:

```bash
nvidia-smi                                           # busy GPU means it is generating
watch -n5 'find ~/cosmos/outputs -name vision.mp4'   # clips as they land
```

The output directory is only created when the first clip is written, so it
staying empty for the first few minutes is expected.

### Telling work from a hang

A silent run is normal for the first several minutes. In order, the causes:

| Check | Meaning |
| --- | --- |
| `ps aux \| grep -E "uv\|pip" \| grep -v grep` | The notebook re-runs its own install cell on every execution, even when the environment already exists. Minutes of silence, nothing wrong. Note this also matches `pipewire`, the audio daemon — ignore those. |
| `nvidia-smi` | A busy GPU means it is generating. |
| `du -sh ~/.cache/huggingface` | Growing means something is still downloading — `cosmos_guardrail` fetches its own model when the pipeline is constructed, before any generation. |
| `ps aux \| grep ipykernel \| grep -v grep` | No kernel process means it died and the runner is wedged. Kill and restart. |

Arrow keys echoing as `^[[A` in the terminal is another sign the foreground
process is busy rather than waiting on input.

### To run it interactively instead

Tunnel a browser in rather than binding to a public interface — the token is the
only authentication:

```bash
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
ssh -N -L 8888:127.0.0.1:8888 <user>@<dgx>     # from the workstation
```

Then switch the kernel to *Cosmos3 Diffusers (Python 3.13)* when the notebook
says to, and run the Restore Environment cell immediately after.

### The five controls

Every asset ships in the repo, about 10 MB in total; nothing is downloaded.

| Control | Asset | Gated? |
| --- | --- | --- |
| Edge (Canny) | `assets/edge/control_edge.mp4` | no |
| Depth | `assets/depth/control_depth.mp4` | no |
| Blur | `assets/blur/control_blur.mp4` | **yes** |
| Segmentation | `assets/seg/control_seg.mp4` | **yes** |
| World scenario map | `assets/wsm/control_wsm.mp4` | **yes** |

Gated controls need access to
[nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail),
which is granted by request. Edge and depth work without it, so start there.

The control videos are precomputed structural signals, not raw footage: the
model takes the signal plus a caption and generates a clip that follows it. The
captions in `prompt.json` are structured scene descriptions — subjects,
lighting, cinematography, per-segment actions — rather than a sentence.

### How long it takes

Transfer is **not** in [`inference_benchmarks.md`](./inference_benchmarks.md);
only t2v, i2v and t2i are, and the step count behind those numbers is not
stated. Scaling the B300 Diffusers i2v figure (139.6 s for 189 frames at 720p)
by this workload's 121 frames and 50 steps puts a 720p clip at roughly **two to
three minutes**, so five controls is 10-15 minutes. Treat that as an
order of magnitude, not a measurement.

Diffusers is the slow path by design — it runs without custom CUDA graphs. On
the same B300, vLLM-Omni does 720p t2v in 102 s and NIM in 90 s against
Diffusers' 139 s. Both need Linux, and vLLM-Omni and SGLang are reachable
through `run_docker.sh`.

## Windows

Only the Diffusers path is reachable; vLLM-Omni, SGLang, TensorRT-LLM and NIM
are Linux and container paths. The notebooks are Linux-only too — they check for
`$VENV/bin/python`, which uv does not create on Windows. Use Windows for reading
the cookbooks and editing prompts, and the DGX to run them.
