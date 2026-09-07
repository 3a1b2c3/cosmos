#!/bin/bash
# Generate a long transfer as independent segments instead of one run.
#
# The framework warns that anything outside [24, 200] frames may degrade, and a
# single long run produces nothing at all if it dies partway -- the video is
# written only after every chunk completes. This cuts the control video into
# segments, generates each one separately, and skips segments that already have
# output, so an interrupted job resumes instead of restarting.
#
#   bash run_segmented_transfer.sh                    # 121-frame segments
#   SEG_FRAMES=200 bash run_segmented_transfer.sh     # the documented maximum
#   LIMIT=3 bash run_segmented_transfer.sh            # only the first three
#   CONCAT=1 bash run_segmented_transfer.sh           # join the results at the end
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFER="$HERE/cookbooks/cosmos3/generator/transfer"
REPO="${COSMOS3_REPO:-$HERE/packages/cosmos3}"
SPEC="${SPEC:-$TRANSFER/specs/seg_custom.json}"
OUT="${OUT:-$TRANSFER/outputs/segmented}"
WORK="${WORK:-$OUT/pieces}"
MODEL="${MODEL:-Cosmos3-Nano}"
SEED="${SEED:-2026}"
# 121 is what every shipped spec uses and so the best-exercised value. 200 is
# the top of the range the framework calls acceptable.
SEG_FRAMES="${SEG_FRAMES:-121}"
# Empty means whatever the spec says, normally 50. Halving it roughly halves
# sampling time, at the cost of fine detail and temporal stability.
STEPS="${STEPS:-}"
GUARDRAILS="${GUARDRAILS:-0}"

export COSMOS3_TRANSFER_ROOT="$TRANSFER"
export HF_HUB_DISABLE_SYMLINKS="${HF_HUB_DISABLE_SYMLINKS:-1}"
unset LD_LIBRARY_PATH

PYTHON="$REPO/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "ERROR: $PYTHON is missing. Run run_framework_headless.sh first." >&2
  exit 1
fi
if [ ! -f "$SPEC" ]; then
  echo "ERROR: $SPEC is missing." >&2
  exit 1
fi
for tool in ffmpeg ffprobe; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: $tool is not on PATH; it is needed to cut the control video." >&2
    exit 1
  fi
done

CONTROL_HINT="$("$PYTHON" -c "
import json, sys
spec = json.load(open(sys.argv[1], encoding='utf-8'))
for key, value in spec.items():
    if isinstance(value, dict) and 'control_path' in value:
        print(key); break
" "$SPEC")"
CONTROL_PATH="$("$PYTHON" -c "
import json, sys
from pathlib import Path
spec_path = Path(sys.argv[1])
spec = json.loads(spec_path.read_text(encoding='utf-8'))
print((spec_path.parent / spec[sys.argv[2]]['control_path']).resolve())
" "$SPEC" "$CONTROL_HINT")"

if [ ! -f "$CONTROL_PATH" ]; then
  echo "ERROR: control '$CONTROL_HINT' points at a missing file: $CONTROL_PATH" >&2
  exit 1
fi

TOTAL="$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$CONTROL_PATH")"
SEGMENTS=$(( (TOTAL + SEG_FRAMES - 1) / SEG_FRAMES ))
if [ -n "${LIMIT:-}" ] && [ "$LIMIT" -lt "$SEGMENTS" ]; then
  SEGMENTS="$LIMIT"
fi

echo "=========================================="
echo "control:  $CONTROL_PATH ($CONTROL_HINT, $TOTAL frames)"
echo "segments: $SEGMENTS x $SEG_FRAMES frames"
echo "out:      $OUT"
echo "=========================================="
echo

mkdir -p "$WORK"
GUARD_FLAG="--no-guardrails"
if [ "$GUARDRAILS" = "1" ]; then
  GUARD_FLAG="--guardrails"
fi

for index in $(seq 0 $((SEGMENTS - 1))); do
  NAME="$(printf 'seg%03d' "$index")"
  FIRST=$((index * SEG_FRAMES))
  LAST=$((FIRST + SEG_FRAMES - 1))
  if [ "$LAST" -ge "$TOTAL" ]; then
    LAST=$((TOTAL - 1))
  fi
  COUNT=$((LAST - FIRST + 1))
  # A final partial segment below the framework's floor would be generated at a
  # length it warns about, for a fraction of a second of output.
  if [ "$COUNT" -lt 24 ]; then
    echo "$NAME: only $COUNT frames left, below the supported minimum of 24. Stopping."
    break
  fi

  SEG_OUT="$OUT/$NAME"
  if find "$SEG_OUT" -name "*.mp4" 2>/dev/null | grep -q .; then
    echo "$NAME: already generated, skipping."
    continue
  fi

  PIECE="$WORK/${NAME}_control.mp4"
  if [ ! -f "$PIECE" ]; then
    # Frame-accurate rather than seconds-based: an off-by-one here would shift
    # the control against the frames the model generates.
    ffmpeg -v error -y -i "$CONTROL_PATH" -vf "select='between(n,$FIRST,$LAST)',setpts=N/FRAME_RATE/TB" -frames:v "$COUNT" -c:v libx264 -crf 18 -pix_fmt yuv420p "$PIECE"
  fi

  PIECE_SPEC="$WORK/${NAME}.json"
  STEPS="$STEPS" "$PYTHON" - "$SPEC" "$PIECE_SPEC" "$PIECE" "$COUNT" "$CONTROL_HINT" <<'PY'
import json
import os
import sys
from pathlib import Path

source, target, piece, frames, hint = sys.argv[1:6]
spec = json.loads(open(source, encoding="utf-8").read())
spec["name"] = f"{spec['name']}_{Path(target).stem}"
spec["num_frames"] = int(frames)
# The chunk size is capped at the segment length so each segment is a single
# pass, with no autoregressive hand-off inside it.
spec["num_video_frames_per_chunk"] = min(int(frames), spec.get("num_video_frames_per_chunk", 121))
if os.environ.get("STEPS"):
    spec["num_steps"] = int(os.environ["STEPS"])
spec[hint]["control_path"] = piece
with open(target, "w", encoding="utf-8") as handle:
    json.dump(spec, handle, indent=2)
PY

  echo "--- $NAME: frames $FIRST-$LAST ($COUNT) ---"
  mkdir -p "$SEG_OUT"
  cd "$REPO"
  .venv/bin/python -m cosmos_framework.scripts.inference $GUARD_FLAG --parallelism-preset=latency -i "$PIECE_SPEC" -o "$SEG_OUT" --checkpoint-path "$MODEL" --seed "$SEED"
  cd "$HERE"
done

echo
echo "=========================================="
echo "Segments generated:"
find "$OUT" -name "*.mp4" -not -path "$WORK/*" -printf "  %TT %10s %p\n" 2>/dev/null | sort || true

if [ "${CONCAT:-0}" = "1" ]; then
  LIST="$WORK/concat.txt"
  : > "$LIST"
  # Sorted by segment name so the pieces join in generation order.
  while IFS= read -r clip; do
    echo "file '$clip'" >> "$LIST"
  done < <(find "$OUT" -name "vision.mp4" -not -path "$WORK/*" | sort)
  if [ -s "$LIST" ]; then
    ffmpeg -v error -y -f concat -safe 0 -i "$LIST" -c copy "$OUT/joined.mp4"
    echo
    echo "joined: $OUT/joined.mp4"
  fi
fi
