#!/usr/bin/env bash
# Start FastAPI backend for ScanBass
set -euo pipefail

PORT_VALUE="$(python -m port_config --default 10000)"

export SCANBASS_PORT="${PORT_VALUE}"
export PORT="${PORT_VALUE}"

echo "Starting ScanBass backend on port ${PORT_VALUE}" >&2
exec uvicorn deploy_backend.src.web_service:app --host 0.0.0.0 --port "${PORT_VALUE}"
