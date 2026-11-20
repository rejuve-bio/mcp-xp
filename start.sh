#!/bin/bash
set -e

# Validate required environment variables
echo "Checking enviroment variables"
for var in APP_PORT MCP_PORT REDIS_PORT QDRANT_HTTP_PORT; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is not set" >&2
        exit 1
    fi
done

# Create logs directory
mkdir -p /app/logs

# Start BioBlend server in background with logging to file + terminal
echo "Starting BioBlend server on port $MCP_PORT..."
uv run python -m app.bioblend_server --host 0.0.0.0 --port "$MCP_PORT" \
    2>&1 | tee /app/logs/MCP_server.log &

# Start Uvicorn app in background with logging to file + terminal
echo "Starting Uvicorn Galaxy Integration service on port $APP_PORT..."
# --proxy-headers: log the real client address that the reverse proxy forwards,
# trusting X-Forwarded-For only from FORWARDED_ALLOW_IPS (the proxy's address).
uv run uvicorn app.main:app --host 0.0.0.0 --port "$APP_PORT" --log-level debug --reload \
    --proxy-headers --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-127.0.0.1}" \
    2>&1 | tee /app/logs/GX_integration.log &

# Trap SIGTERM and clean up background processes gracefully
trap 'echo "Stopping services..."; kill $(jobs -p)' SIGTERM

# Wait for any process to exit
wait -n

# Exit with the status of the first failed process
exit $?
