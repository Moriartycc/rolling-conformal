#!/usr/bin/env bash

set -euo pipefail

# Paper configuration. Extra command-line arguments override these values.
MAX_N=5000
D=200
SIGMA=0.2
TRIALS=100
CHECKPOINT_STEP=20
SEED=2026
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
    --max-n "${MAX_N}"
    --d "${D}"
    --sigma "${SIGMA}"
    --trials "${TRIALS}"
    --checkpoint-step "${CHECKPOINT_STEP}"
    --seed "${SEED}"
    --output-dir "${RESOLVED_OUTPUT_DIR}"
)

echo "Running the RoCP versus split-conformal OLS experiment:"
echo "  max n=${MAX_N}, d=${D}, sigma=${SIGMA}, M=${TRIALS}"
echo "  checkpoint step=${CHECKPOINT_STEP}, seed=${SEED}"
echo "  output=${RESOLVED_OUTPUT_DIR}"

"${PYTHON}" "${args[@]}" "$@"
