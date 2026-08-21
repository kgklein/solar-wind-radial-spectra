#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_executable="${repository_root}/.venv/bin/python"
export SPEDAS_DATA_DIR="${repository_root}/data"

if [[ ! -x "${python_executable}" ]]; then
    echo "The project virtual environment was not found at ${repository_root}/.venv." >&2
    echo "Create it and install the project with:" >&2
    echo "  python3.11 -m venv .venv" >&2
    echo "  .venv/bin/python -m pip install -e '.[test]'" >&2
    exit 1
fi

cd "${repository_root}"
exec "${python_executable}" scripts/plot_interval.py \
    --mission PSP \
    --start 2022-02-25T00:00:00 \
    --stop 2022-02-26T00:00:00 \
    "$@"
