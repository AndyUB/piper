#!/bin/bash
# Analyze Piper nsys traces: export .nsys-rep → SQLite, then run Python analysis.
#
# Usage:
#   ./analyze_nsys.sh <trace_dir>          # analyze all .nsys-rep in the directory
#   ./analyze_nsys.sh <file1.nsys-rep> ... # analyze specific files
#
# Options (via env vars):
#   NSYS_BIN      path to nsys binary (default: auto-detected)
#   WARMUP        warmup iterations to exclude (default: 2)
#   ITERS         timed iterations to analyze (default: 5)
#   TRACE_ITERS   tracing iterations to exclude (default: 3)
#   TOPN          number of top kernels to show (default: 15)
#   OUT_DIR       directory for output files (default: analysis_out/)
#   KEEP_DB       set to 1 to keep the exported .sqlite files
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── locate nsys ──
if [[ -n "${NSYS_BIN:-}" ]]; then
    NSYS="$NSYS_BIN"
elif command -v nsys &>/dev/null; then
    NSYS="$(command -v nsys)"
elif [[ -x /opt/nvidia/nsight-systems/2025.1.3/bin/nsys ]]; then
    NSYS=/opt/nvidia/nsight-systems/2025.1.3/bin/nsys
else
    echo "ERROR: nsys not found. Set NSYS_BIN or add nsys to PATH." >&2
    exit 1
fi
echo "Using nsys: $NSYS ($("$NSYS" --version 2>&1 | head -1))"

# ── collect .nsys-rep files ──
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <trace_dir_or_nsys-rep_files...>" >&2
    exit 1
fi

REP_FILES=()
LABEL="${LABEL:-}"

for arg in "$@"; do
    if [[ -d "$arg" ]]; then
        # derive label from directory name if not set
        if [[ -z "$LABEL" ]]; then
            LABEL="$(basename "$arg")"
        fi
        while IFS= read -r -d '' f; do
            REP_FILES+=("$f")
        done < <(find "$arg" -name "*.nsys-rep" -print0 2>/dev/null | sort -z)
    elif [[ -f "$arg" && "$arg" == *.nsys-rep ]]; then
        REP_FILES+=("$arg")
    else
        echo "WARN: skipping '$arg' (not a directory or .nsys-rep file)" >&2
    fi
done

if [[ ${#REP_FILES[@]} -eq 0 ]]; then
    echo "ERROR: no .nsys-rep files found." >&2
    exit 1
fi
echo "Found ${#REP_FILES[@]} trace file(s) | label: ${LABEL:-piper}"

# ── export each .nsys-rep → .sqlite ──
TMP_DIR=$(mktemp -d /tmp/piper_nsys_XXXXXX)
trap '[[ "${KEEP_DB:-0}" == "1" ]] || rm -rf "$TMP_DIR"' EXIT

SQLITE_FILES=()
for rep in "${REP_FILES[@]}"; do
    base="$(basename "$rep" .nsys-rep)"
    out="$TMP_DIR/${base}.sqlite"
    printf "Exporting %-45s ... " "$(basename "$rep")"
    "$NSYS" export --type sqlite --output "$out" "$rep" 2>/dev/null
    if [[ -f "$out" ]]; then
        SQLITE_FILES+=("$out")
        echo "ok"
    else
        echo "FAILED"
    fi
done

if [[ ${#SQLITE_FILES[@]} -eq 0 ]]; then
    echo "ERROR: no SQLite files produced." >&2
    exit 1
fi

[[ "${KEEP_DB:-0}" == "1" ]] && echo "SQLite files kept in: $TMP_DIR"

# ── run Python analysis ──
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/analysis_out}"
mkdir -p "$OUT_DIR"
echo ""
echo "Running analysis (warmup=${WARMUP:-2}, iters=${ITERS:-5}, trace=${TRACE_ITERS:-3}) ..."
python3 "$SCRIPT_DIR/analyze_nsys.py" \
    --warmup      "${WARMUP:-2}" \
    --iters       "${ITERS:-5}" \
    --trace-iters "${TRACE_ITERS:-3}" \
    --topn        "${TOPN:-15}" \
    --out-dir     "$OUT_DIR" \
    --label       "${LABEL:-piper}" \
    "${SQLITE_FILES[@]}"
