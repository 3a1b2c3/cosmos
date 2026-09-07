#!/bin/bash
# Turn the checked-in custom source clip into a segmentation control video next
# to it, ready for specs/seg_custom.json.
#
#   bash convert.sh                        # full source, automatic prompting
#   FRAMES=121 bash convert.sh             # one spec-length clip instead
#   POINTS="960,700" bash convert.sh       # prompt one object explicitly
#
# Run setup.sh first.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFER="$HERE/../cookbooks/cosmos3/generator/transfer"
PY="$HERE/.venv/bin/python"

SOURCE="${SOURCE:-$TRANSFER/assets/custom/ePi8tDdKuWw.mp4}"
OUT="${OUT:-$TRANSFER/assets/custom/control_seg.mp4}"
# 0 means the whole source. The spec's own num_frames has to match whatever this
# produces, so changing one without the other desynchronises them.
FRAMES="${FRAMES:-0}"
MAXOBJ="${MAXOBJ:-14}"

if [ ! -x "$PY" ]; then
  echo "ERROR: $PY is missing. Run setup.sh first." >&2
  exit 1
fi
if [ ! -f "$SOURCE" ]; then
  echo "ERROR: $SOURCE is missing." >&2
  exit 1
fi

echo "=========================================="
echo "source: $SOURCE"
echo "out:    $OUT"
echo "frames: $FRAMES  (0 = whole source)"
echo "=========================================="
echo

if [ -n "${POINTS:-}" ]; then
  "$PY" "$HERE/make_seg_control.py" "$SOURCE" -o "$OUT" --frames "$FRAMES" --max-objects "$MAXOBJ" --points "$POINTS"
else
  "$PY" "$HERE/make_seg_control.py" "$SOURCE" -o "$OUT" --frames "$FRAMES" --max-objects "$MAXOBJ"
fi

echo
echo "=========================================="
echo "Next: set num_frames in specs/seg_custom.json to the frame count above,"
echo "      then run the transfer notebook with that spec."
echo "=========================================="
