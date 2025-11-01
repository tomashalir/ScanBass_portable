#!/usr/bin/env bash
# Start FastAPI backend for ScanBass

set -euo pipefail

# Render exposes the port to bind through the PORT variable. Fall back to the
# legacy SCANBASS_PORT knob (used locally) and only finally to Render's default
# port to preserve backwards compatibility when no configuration is provided.
PORT_VALUE="${PORT:-}"
if [[ -z "${PORT_VALUE}" ]]; then
    PORT_VALUE="${SCANBASS_PORT:-}"
fi
if [[ -z "${PORT_VALUE}" ]]; then
    PORT_VALUE="10000"
fi

export PORT="${PORT_VALUE}"
export SCANBASS_PORT="${PORT_VALUE}"

echo "Starting ScanBass backend on port ${PORT_VALUE}"

exec python -m uvicorn deploy_backend.src.web_service:app --host 0.0.0.0 --port "${PORT_VALUE}"
