#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_directory}/.." && pwd)"
environment_directory="${repository_root}/.venv-feynkit"
marimo_executable="${environment_directory}/bin/marimo"
node_directory="${environment_directory}/nodejs/bin"

export PATH="${environment_directory}/bin:${node_directory}:${PATH}"

if [[ ! -x "${marimo_executable}" ]]; then
    echo "Marimo is not installed in ${environment_directory}." >&2
    echo "Create the environment and install marimo and ty before starting the server." >&2
    exit 1
fi

if ! command -v node >/dev/null 2>&1; then
    echo "Node.js is required by Marimo's language-server bridge." >&2
    exit 1
fi

if ! command -v ty >/dev/null 2>&1; then
    echo "ty is required by the project Marimo configuration." >&2
    exit 1
fi

marimo_host="${MARIMO_HOST:-0.0.0.0}"
marimo_port="${MARIMO_PORT:-2718}"
notebook_path="${MARIMO_NOTEBOOK_PATH:-${repository_root}/examples/feynkit}"
fixed_access_token="6jr3QPRRx8gxWnt1c5X2Jg"

cd "${repository_root}"

exec "${marimo_executable}" edit "${notebook_path}" \
    --headless \
    --host "${marimo_host}" \
    --port "${marimo_port}" \
    --skip-update-check \
    --token-password-file - <<<"${fixed_access_token}"
