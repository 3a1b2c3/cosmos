#!/bin/bash
# Turn the checked-in custom source clip into a Cityscapes-class segmentation
# control video next to it, ready for specs/seg_custom.json.
#
# The counterpart of convert.sh, which uses SAM 2. This one labels every pixel
# by class, so a colour means the same thing in every frame of every video
# rather than "the third-largest region in this shot".
#
#   bash convert_semantic.sh                  # full source
#   FRAMES=300 bash convert_semantic.sh       # a shorter test
#   BATCH=8 bash convert_semantic.sh          # more frames per forward pass
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
BATCH="${BATCH:-4}"

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
echo "mode:   Cityscapes semantic classes"
echo "=========================================="
echo

"$PY" "$HERE/make_seg_control_semantic.py" "$SOURCE" -o "$OUT" --frames "$FRAMES" --batch "$BATCH"

# Checked explicitly: a crash inside the loop has been seen to leave the exit
# code at zero while writing no video at all.
if [ ! -f "$OUT" ]; then
  echo "ERROR: the run reported success but $OUT was not written." >&2
  exit 1
fi

echo
echo "=========================================="
echo "Next: set num_frames in specs/seg_custom.json to the frame count above,"
echo "      then run the transfer with that spec."
echo "=========================================="
