#!/usr/bin/env bash
set -e

HOST=0.0.0.0
PORT=${PORT:-8000}

exec uvicorn deploy_backend.web_service:app --host "$HOST" --port "$PORT"
