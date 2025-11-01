#!/usr/bin/env bash
set -e

# Render dává PORT do env, lokálně to bude 8000
HOST=0.0.0.0
PORT=${PORT:-8000}

exec uvicorn web_service:app --host "$HOST" --port "$PORT"


