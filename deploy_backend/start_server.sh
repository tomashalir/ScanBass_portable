#!/usr/bin/env bash
# Start FastAPI backend for ScanBass

set -euo pipefail

PORT_VALUE=${PORT:-8000}

exec python -m uvicorn deploy_backend.src.web_service:app --host 0.0.0.0 --port "${PORT_VALUE}"
