#!/usr/bin/env bash
# Render every .tex here to docs/src/assets/*.svg.
# Only needed when a diagram changes — the SVGs are committed so CI needs no LaTeX.
set -euo pipefail
cd "$(dirname "$0")"
out=../src/assets
mkdir -p "$out" build
for f in fig_*.tex; do
  b="${f%.tex}"
  pdflatex -interaction=batchmode -halt-on-error -output-directory=build "$f" >/dev/null
  pdf2svg "build/$b.pdf" "$out/$b.svg"
  echo "  $out/$b.svg"
done
rm -rf build
