#!/usr/bin/env bash
# dhwani — one command. Sets everything up, measures everything, packs the results.
#
#     cd <this folder>
#     bash runall.sh
#
# Safe to stop (Ctrl-C) and re-run: finished work is skipped, not repeated.
#
#   SMOKE=1 bash runall.sh          ~5 minutes, tiny models — proves it all works
#   QUICK=1 bash runall.sh          ~45 minutes instead of ~3 hours
#   MACBENCH_MAX_HOURS=2 bash ...   stop starting new work after N hours
#   SKIP_SETUP=1 bash runall.sh     reuse the venv and models already here

set -uo pipefail          # deliberately NOT -e: a failed step must not end the run
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."   # repo root; this script lives in macbench/

VENV=".venv-macbench"
PY="$VENV/bin/python"
RESULTS="macbench/results"
mkdir -p "$RESULTS"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[33m!!  %s\033[0m\n' "$*"; }

step "dhwani measurement run — started $(date)"
cat <<'INTRO'

  What this does, in order:
    1. builds a private python environment in .venv-macbench   (~2 min)
    2. downloads the speech models into the huggingface cache  (~5 GB, 5-30 min
       depending on the connection; more later if the model-comparison rows run)
    3. runs the test suite
    4. checks the sealed server survives the sandbox the real scoring uses
    5. measures ~30 engine configurations               (this is the long part)
    6. writes macbench/results/REPORT.md and a zip to send back

  It needs about 25 GB free and leaves nothing outside this folder except the
  huggingface model cache. Nothing is uploaded anywhere.

INTRO

# --- 1. environment --------------------------------------------------------
if [ "${SKIP_SETUP:-0}" != "1" ]; then
  step "1/6  python environment"

  PYBIN=""
  for c in python3.12 python3.11 python3.13 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)' 2>/dev/null; then
      PYBIN="$c"; break
    fi
  done
  if [ -z "$PYBIN" ]; then
    warn "No Python 3.10+ found. Install one and re-run:"
    echo "      brew install python@3.12        (or download from python.org)"
    exit 1
  fi
  echo "    using $PYBIN ($("$PYBIN" --version 2>&1))"

  [ -d "$VENV" ] || "$PYBIN" -m venv "$VENV"
  "$PY" -m pip install --quiet --upgrade pip wheel
  echo "    installing dependencies (a few minutes the first time)"
  "$PY" -m pip install --quiet -r requirements.txt -r requirements-streaming.txt \
      || warn "some requirements failed — the run continues and will record what broke"
  "$PY" -m pip install --quiet pytest numpy soundfile websockets \
      || warn "pytest/numpy install failed"

  # mlx-whisper is the Apple-GPU path and is the whole point of running here.
  if [ "$(uname -m)" = "arm64" ]; then
    "$PY" -m pip install --quiet "mlx-whisper>=0.4" || warn "mlx-whisper failed to install — the run will fall back to CPU and the numbers will not represent the scoring box"
  else
    warn "this is not an Apple-silicon Mac ($(uname -m)); the GPU backend cannot be measured here"
  fi
else
  step "1/6  reusing existing environment (SKIP_SETUP=1)"
fi

[ -x "$PY" ] || { warn "no python at $PY — cannot continue"; exit 1; }

# --- 2. models -------------------------------------------------------------
step "2/6  downloading speech models into the huggingface cache"
echo "    (about 5 GB for the shipped configuration; comparison rows fetch more"
echo "     as they run, and record an error instead of stopping if one fails)"
"$PY" - <<'PYWARM' || warn "model warm-up hit a problem — recorded, continuing"
import time, sys
sys.path.insert(0, ".")
t = time.monotonic()
from solution.draft import warm_models
warm_models()
print(f"    models ready in {time.monotonic()-t:.0f}s")
PYWARM

# --- 3. audio --------------------------------------------------------------
step "3/6  checking the audio corpus"
n=$(ls data/dev/audio/*.wav 2>/dev/null | wc -l | tr -d ' ')
echo "    $n clips in data/dev/audio, $(ls samples/*.wav 2>/dev/null | wc -l | tr -d ' ') in samples/"
if [ "$n" -lt 5 ]; then
  warn "the corpus audio is missing — trying to fetch FLEURS (needs network)"
  "$PY" scripts/fetch_audio.py --manifest data/dev/manifest.json --out data/dev/audio \
      || warn "fetch failed; the run will measure whatever clips are present"
fi

# --- 4/5. the battery ------------------------------------------------------
step "4/6  measuring — this is the long part"
echo "    progress prints as it goes; results land in $RESULTS as each row finishes."
echo "    Interrupting and re-running this script resumes where it stopped."
PYTHONPATH="$PWD/macbench" "$PY" -m bench.drive
rc=$?
[ $rc -eq 0 ] || warn "the battery exited with status $rc — whatever finished is still in $RESULTS"

# --- 6. pack ---------------------------------------------------------------
step "5/6  writing the report"
PYTHONPATH="$PWD/macbench" "$PY" -m bench.report || warn "report generation failed; the raw JSON is still there"

step "6/6  packing the results"
STAMP=$(date +%Y%m%d-%H%M)
OUT="dhwani-results-$STAMP.zip"
rm -f "$OUT"
zip -qr "$OUT" "$RESULTS" && echo "    wrote $(pwd)/$OUT"

cat <<EOF

  ------------------------------------------------------------------
  DONE.

  Send back:   $(pwd)/$OUT

  If sending a zip is awkward, the short version is in:
               $(pwd)/$RESULTS/SUMMARY.txt
  and the full write-up is in:
               $(pwd)/$RESULTS/REPORT.md
  ------------------------------------------------------------------
EOF
