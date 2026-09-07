#!/bin/bash
# Run a transfer against a custom spec through the Cosmos Framework entrypoint,
# rather than through the notebook.
#
# The notebook's cells hardcode the shipped specs, so a custom control video is
# not reachable from it. This calls the same entrypoint the notebook's %%bash
# cells call, with the spec pointed wherever you want.
#
#   bash run_custom_transfer.sh                          # specs/seg_custom.json
#   SPEC=specs/edge.json bash run_custom_transfer.sh     # a different spec
#   MODEL=Cosmos3-Super bash run_custom_transfer.sh      # 32B, multi-GPU
#   GUARDRAILS=1 bash run_custom_transfer.sh             # needs Cosmos-Guardrail1
#   FRAMES=121 bash run_custom_transfer.sh               # override num_frames
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFER="$HERE/cookbooks/cosmos3/generator/transfer"
REPO="${COSMOS3_REPO:-$HERE/packages/cosmos3}"
SPEC="${SPEC:-$TRANSFER/specs/seg_custom.json}"
OUT="${OUT:-$TRANSFER/outputs/custom}"
MODEL="${MODEL:-Cosmos3-Nano}"
SEED="${SEED:-2026}"
# Inference defaults to --guardrails, which downloads nvidia/Cosmos-Guardrail1.
# That repository is gated separately from nvidia/Cosmos-1.0-Guardrail and the
# framework fetches it for every control, so without access the run fails before
# producing anything.
GUARDRAILS="${GUARDRAILS:-0}"

# The notebook's §2 cell exports these; running the entrypoint directly means
# setting them here. LD_LIBRARY_PATH is cleared because the NGC images put a
# bundled libtorch on it that conflicts with the venv's own.
export COSMOS3_TRANSFER_ROOT="$TRANSFER"
export HF_HUB_DISABLE_SYMLINKS="${HF_HUB_DISABLE_SYMLINKS:-1}"
unset LD_LIBRARY_PATH

if [ ! -x "$REPO/.venv/bin/python" ]; then
  echo "ERROR: $REPO/.venv/bin/python is missing." >&2
  echo "  Run run_framework_headless.sh first, or set COSMOS3_REPO." >&2
  exit 1
fi
if [ ! -f "$SPEC" ]; then
  echo "ERROR: $SPEC is missing." >&2
  exit 1
fi

# Resolve the control videos the spec names and check they exist. A missing
# control otherwise surfaces deep inside the pipeline as something less obvious.
CONTROLS="$("$REPO/.venv/bin/python" - "$SPEC" <<'PY'
import json
import sys
from pathlib import Path

spec_path = Path(sys.argv[1])
spec = json.loads(spec_path.read_text(encoding="utf-8"))
for key, value in spec.items():
    if isinstance(value, dict) and "control_path" in value:
        print(key, (spec_path.parent / value["control_path"]).resolve(), sep="\t")
PY
)"
if [ -z "$CONTROLS" ]; then
  echo "ERROR: $SPEC names no control videos." >&2
  exit 1
fi
while IFS=$'\t' read -r hint path; do
  if [ ! -f "$path" ]; then
    echo "ERROR: control '$hint' points at a missing file:" >&2
    echo "  $path" >&2
    exit 1
  fi
  echo "control $hint: $path"
done <<< "$CONTROLS"

# num_frames has to match the control video, and the two are edited in different
# places -- the spec by hand, the video by segmap. A mismatch is the most likely
# way this run goes wrong, so it is checked rather than discovered later.
if command -v ffprobe >/dev/null 2>&1; then
  SPEC_FRAMES="$("$REPO/.venv/bin/python" -c "import json,sys; print(json.load(open(sys.argv[1]))['num_frames'])" "$SPEC")"
  while IFS=$'\t' read -r hint path; do
    ACTUAL="$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$path" 2>/dev/null || echo "")"
    if [ -n "$ACTUAL" ] && [ "$ACTUAL" != "$SPEC_FRAMES" ]; then
      echo >&2
      echo "ERROR: spec num_frames=$SPEC_FRAMES but '$hint' has $ACTUAL frames." >&2
      echo "  Edit $SPEC, or re-run with FRAMES=$ACTUAL to override it." >&2
      exit 1
    fi
  done <<< "$CONTROLS"
  echo "num_frames: $SPEC_FRAMES, matching the control video"
fi

RUN_SPEC="$SPEC"
if [ -n "${FRAMES:-}" ] || [ -n "${STEPS:-}" ]; then
  # Written beside the original rather than in place, so the checked-in spec
  # keeps its own values.
  RUN_SPEC="${SPEC%.json}.run.json"
  FRAMES="${FRAMES:-}" STEPS="${STEPS:-}" "$REPO/.venv/bin/python" - "$SPEC" "$RUN_SPEC" <<'PY'
import json
import os
import sys

source, target = sys.argv[1], sys.argv[2]
spec = json.loads(open(source, encoding="utf-8").read())
if os.environ["FRAMES"]:
    spec["num_frames"] = int(os.environ["FRAMES"])
    print(f"  num_frames overridden to {spec['num_frames']}")
if os.environ["STEPS"]:
    # Halving num_steps roughly halves sampling time. UniPC holds up better
    # than most samplers at low step counts, but this is a quality trade, not
    # a free one -- fine detail and temporal stability go first.
    spec["num_steps"] = int(os.environ["STEPS"])
    print(f"  num_steps overridden to {spec['num_steps']}")
with open(target, "w", encoding="utf-8") as handle:
    json.dump(spec, handle, indent=2)
PY
fi

mkdir -p "$OUT"
echo "=========================================="
echo "spec:   $RUN_SPEC"
echo "model:  $MODEL"
echo "out:    $OUT"
echo "guard:  $([ "$GUARDRAILS" = "1" ] && echo enabled || echo disabled)"
echo "=========================================="
echo

GUARD_FLAG="--no-guardrails"
if [ "$GUARDRAILS" = "1" ]; then
  GUARD_FLAG="--guardrails"
fi

# EXTRA_ARGS is passed through verbatim, for entrypoint flags this script does
# not model -- disabling the diffusion cache, for instance, which trades speed
# for fidelity and is on by default.
cd "$REPO"
.venv/bin/python -m cosmos_framework.scripts.inference $GUARD_FLAG --parallelism-preset=latency -i "$RUN_SPEC" -o "$OUT" --checkpoint-path "$MODEL" --seed "$SEED" ${EXTRA_ARGS:-}

echo
echo "=========================================="
echo "Done. Generated clips:"
echo "=========================================="
find "$OUT" -name "*.mp4" -printf "  %TT %10s %p\n" 2>/dev/null | sort || echo "  none found under $OUT"
