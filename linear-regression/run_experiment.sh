#!/usr/bin/env bash

set -euo pipefail

# Paper configuration. Extra command-line arguments override these values.
N=40000
D=200
SIGMA=1.0
TRIALS=100
SEED=2026
HOLDOUT_SIZE=500
SET_NOMINAL_COVERAGE=0.90
Y_GRID_SIZE=401
OUTPUT_DIR="output"

# Leave empty for automatic detection, or provide an explicit executable.
PYTHON=""

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python_works() {
    [[ -x "$1" ]] && "$1" -c "import sys" >/dev/null 2>&1
}

if [[ -z "${PYTHON}" ]]; then
    if python_works "${SCRIPT_DIR}/.venv/Scripts/python.exe"; then
        PYTHON="${SCRIPT_DIR}/.venv/Scripts/python.exe"
    elif python_works "${SCRIPT_DIR}/.venv/bin/python"; then
        PYTHON="${SCRIPT_DIR}/.venv/bin/python"
    elif python_works "${SCRIPT_DIR}/../.venv/Scripts/python.exe"; then
        PYTHON="${SCRIPT_DIR}/../.venv/Scripts/python.exe"
    elif python_works "${SCRIPT_DIR}/../.venv/bin/python"; then
        PYTHON="${SCRIPT_DIR}/../.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1 && \
        python_works "$(command -v python3)"; then
        PYTHON="$(command -v python3)"
    elif command -v python >/dev/null 2>&1 && \
        python_works "$(command -v python)"; then
        PYTHON="$(command -v python)"
    else
        echo "Error: no working Python executable was found." >&2
        exit 1
    fi
fi

if [[ "${OUTPUT_DIR}" = /* ]] || [[ "${OUTPUT_DIR}" =~ ^[A-Za-z]:[/\\] ]]; then
    RESOLVED_OUTPUT_DIR="${OUTPUT_DIR}"
else
    RESOLVED_OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_DIR}"
fi

args=(
    "${SCRIPT_DIR}/main.py"
    --n "${N}"
    --d "${D}"
    --sigma "${SIGMA}"
    --trials "${TRIALS}"
    --seed "${SEED}"
    --holdout-size "${HOLDOUT_SIZE}"
    --set-nominal-coverage "${SET_NOMINAL_COVERAGE}"
    --y-grid-size "${Y_GRID_SIZE}"
    --output-dir "${RESOLVED_OUTPUT_DIR}"
)

echo "Running minimum-norm linear-regression RoCP experiment:"
echo "  n=${N}, d=${D}, sigma=${SIGMA}, M=${TRIALS}, seed=${SEED}"
echo "  holdout size=${HOLDOUT_SIZE}, set coverage=${SET_NOMINAL_COVERAGE}"
echo "  output=${RESOLVED_OUTPUT_DIR}"

"${PYTHON}" "${args[@]}" "$@"
