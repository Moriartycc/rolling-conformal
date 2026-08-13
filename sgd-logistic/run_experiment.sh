#!/usr/bin/env bash

set -euo pipefail

# Paper configuration. Extra command-line arguments override these values.
N=10000
D=10
TRIALS=100
HOLDOUT_SIZE=500
SEED=2026
ETA0=1.0
T0=10.0
GAMMAS="0.6,0.8,1"
MEMBERSHIP_ALPHA=0.3
MARGIN_WINDOW=100

PYTHON=""

OUTPUT_DIR="output"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python_works() {
    [[ -x "$1" ]] && "$1" -c "import sys" >/dev/null 2>&1
}

if [[ -z "${PYTHON}" ]]; then
    if python_works "${SCRIPT_DIR}/.venv/Scripts/python.exe"; then
        # Windows virtual environment used from Git Bash.
        PYTHON="${SCRIPT_DIR}/.venv/Scripts/python.exe"
    elif python_works "${SCRIPT_DIR}/.venv/bin/python"; then
        # Linux, macOS, or WSL virtual environment.
        PYTHON="${SCRIPT_DIR}/.venv/bin/python"
    elif python_works "${SCRIPT_DIR}/../.venv/Scripts/python.exe"; then
        # Shared Windows environment at the Rolling conformal project root.
        PYTHON="${SCRIPT_DIR}/../.venv/Scripts/python.exe"
    elif python_works "${SCRIPT_DIR}/../.venv/bin/python"; then
        # Shared Linux/macOS environment at the project root.
        PYTHON="${SCRIPT_DIR}/../.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1 && \
        python_works "$(command -v python3)"; then
        PYTHON="$(command -v python3)"
    elif command -v python >/dev/null 2>&1 && \
        python_works "$(command -v python)"; then
        PYTHON="$(command -v python)"
    else
        echo "Error: no Python executable was found." >&2
        echo "Create .venv or set PYTHON near the top of this script." >&2
        exit 1
    fi
fi

if ! "${PYTHON}" -c "import sys" >/dev/null 2>&1; then
    echo "Error: configured Python executable does not run: ${PYTHON}" >&2
    exit 1
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
    --trials "${TRIALS}"
    --holdout-size "${HOLDOUT_SIZE}"
    --seed "${SEED}"
    --eta0 "${ETA0}"
    --t0 "${T0}"
    --gammas "${GAMMAS}"
    --membership-alpha "${MEMBERSHIP_ALPHA}"
    --margin-window "${MARGIN_WINDOW}"
    --output-dir "${RESOLVED_OUTPUT_DIR}"
)

echo "Running RoCP experiment with:"
echo "  n=${N}, d=${D}, trials=${TRIALS}, holdout_size=${HOLDOUT_SIZE}, seed=${SEED}"
echo "  eta0=${ETA0}, t0=${T0}, gammas=${GAMMAS}, membership_alpha=${MEMBERSHIP_ALPHA}"
echo "  margin_window=${MARGIN_WINDOW}"
echo "  output=${RESOLVED_OUTPUT_DIR}"

"${PYTHON}" "${args[@]}" "$@"
