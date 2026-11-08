#!/usr/bin/env bash
# Compile the Chinese CVPR working draft.
# Prefer a system XeLaTeX + BibTeX. If those binaries are absent, use the
# Tectonic binary recorded in delivery/verification.md. Template files are not
# modified by this script.
set -euo pipefail
cd -- "$(dirname -- "$0")"
if command -v xelatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  xelatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  xelatex -interaction=nonstopmode -halt-on-error main.tex
  xelatex -interaction=nonstopmode -halt-on-error main.tex
else
  TECTONIC="${TECTONIC:-/tmp/tectonic/tectonic}"
  if [[ ! -x "$TECTONIC" ]]; then
    echo "xelatex/bibtex not found and \$TECTONIC is not executable" >&2
    exit 1
  fi
  "$TECTONIC" -X compile --keep-logs --keep-intermediates main.tex
fi
