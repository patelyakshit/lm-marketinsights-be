#!/bin/sh

# Use PORT from environment (Railway sets this) or default to 8000
PORT=${PORT:-8000}

uvicorn "main:app" \
  --host 0.0.0.0 \
  --port $PORT \
  --workers 4 \
  --loop uvloop \
  --timeout-keep-alive 300 \
  --timeout-graceful-shutdown 30 \
  --backlog 2048 \
  --limit-concurrency 1000 \
  --limit-max-requests 10000 \
  --log-level info \
  --access-log
