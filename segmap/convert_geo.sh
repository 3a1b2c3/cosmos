#!/bin/bash
# Turn the checked-in custom source clip into an edge or depth control video.
#
# Geometric controls describe shape, not meaning, so there are no classes to
# misidentify. That is why they suit stylised or synthetic footage, where
# semantic segmentation fails.
#
#   bash convert_geo.sh                    # edge, full source
#   CONTROL=depth bash convert_geo.sh      # depth instead
#   FRAMES=300 bash convert_geo.sh         # a shorter test
#
# Run setup.sh first.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFER="$HERE/../cookbooks/cosmos3/generator/transfer"
PY="$HERE/.venv/bin/python"

CONTROL="${CONTROL:-edge}"
SOURCE="${SOURCE:-$TRANSFER/assets/custom/ePi8tDdKuWw.mp4}"
# Named per control so edge and depth do not overwrite one another.
OUT="${OUT:-$TRANSFER/assets/custom/control_$CONTROL.mp4}"
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
echo "source:  $SOURCE"
echo "out:     $OUT"
echo "control: $CONTROL"
echo "frames:  $FRAMES  (0 = whole source)"
echo "=========================================="
echo

"$PY" "$HERE/make_geo_control.py" "$SOURCE" -o "$OUT" --control "$CONTROL" --frames "$FRAMES" --batch "$BATCH"

# Checked explicitly: a crash inside the loop has been seen to leave the exit
# code at zero while writing no video at all.
if [ ! -f "$OUT" ]; then
  echo "ERROR: the run reported success but $OUT was not written." >&2
  exit 1
fi

echo
echo "=========================================="
echo "Next: point a spec at $OUT under the \"$CONTROL\" key, set its num_frames"
echo "      to the count above, then run the transfer with that spec."
echo "=========================================="
