#!/usr/bin/env bash

set -euo pipefail

N=60000
M=1000
SEED=2026
ETA0=50
T0=5000
GAMMA=1
DEVICE="auto"
OUTPUT_DIR="output"
DATA_DIR="data"
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
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON="$(command -v python)"
    else
        echo "Error: no working Python executable was found." >&2
        exit 1
    fi
fi

"${PYTHON}" "${SCRIPT_DIR}/main.py" \
    --n "${N}" \
    --M "${M}" \
    --seed "${SEED}" \
    --eta0 "${ETA0}" \
    --t0 "${T0}" \
    --gamma "${GAMMA}" \
    --device "${DEVICE}" \
    --output-dir "${SCRIPT_DIR}/${OUTPUT_DIR}" \
    --data-dir "${SCRIPT_DIR}/${DATA_DIR}" \
    "$@"
