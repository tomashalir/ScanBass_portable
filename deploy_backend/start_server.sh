#!/usr/bin/env bash
# Start FastAPI backend for ScanBass

PORT_VALUE="${SCANBASS_PORT:-}" 
if [[ -z "${PORT_VALUE}" ]]; then
    PORT_VALUE="${PORT:-}"
fi
if [[ -z "${PORT_VALUE}" ]]; then
    PORT_VALUE="10000"
fi

export SCANBASS_PORT="${PORT_VALUE}"
export PORT="${PORT_VALUE}"

python -m uvicorn src.web_service:app --host 0.0.0.0 --port "${PORT_VALUE}"
