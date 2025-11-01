#!/usr/bin/env bash
# Start FastAPI backend for ScanBass
python -m uvicorn src.web_service:app --host 0.0.0.0 --port $PORT
