#!/bin/sh
# Export the diagrams of this folder to docs/src/assets/diagrams/ with the web export of the
# author's website diagram kit (KIT = its tools/diagrams folder): style check, draw.io -> cropped
# PDF (draw.io typesets the TeX), pdf2svg.py -> self-contained SVG with a white background.
# Page 1 of NAME.drawio -> NAME.svg; the page "compact" -> NAME-compact.svg (narrow screens).
#
#   python3 docs/diagrams/portfolioopt--formulation.py && sh docs/diagrams/export.sh
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
KIT=${KIT:-$HOME/Workspace/AndrewRosemberg/tools/diagrams}
OUT="$HERE/../src/assets/diagrams"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/diagrams.XXXXXX")
trap 'rm -rf "$TMP"' EXIT INT TERM
mkdir -p "$OUT"
for f in "$HERE"/*.drawio; do
  name=$(basename "$f" .drawio)
  python3 "$KIT/check_style.py" "$f"
  python3 "$KIT/padframe.py" "$f" "$TMP/padded.drawio"
  drawio -x -f pdf --crop -p 1 -o "$TMP/p1.pdf" "$TMP/padded.drawio" >/dev/null 2>&1
  python3 "$KIT/pdf2svg.py" "$TMP/p1.pdf" "$OUT/$name.svg"
  drawio -x -f pdf --crop -p 2 -o "$TMP/p2.pdf" "$TMP/padded.drawio" >/dev/null 2>&1
  python3 "$KIT/pdf2svg.py" "$TMP/p2.pdf" "$OUT/$name-compact.svg"
done
