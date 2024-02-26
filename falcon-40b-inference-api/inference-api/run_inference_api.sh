#!/bin/bash
# Note: running multiple processes within a single container is NOT best practice
# for docker containerization and kubernetes.

set -eu

test -n "${MODEL_WEKA_DIR}" || (echo "MODEL_WEKA_DIR must be set" && false)
test -n "${JWT_SECRET}" || (echo "JWT_SECRET must be set" && false)

# Get current date and time
START_DATETIME=$(date '+%Y-%m-%d-%H_%M_%S')
# Define the log file location
SERVER_LOG_FILE="${MODEL_WEKA_DIR}/logs/inference_api_server_${START_DATETIME}.log"
PROXY_LOG_FILE="${MODEL_WEKA_DIR}/logs/proxy_${START_DATETIME}.log"

echo "starting proxy, logging output to: ${PROXY_LOG_FILE}"


if [[ -n "${MOCK_MODEL}" ]]; then
    echo "WARNING: this is a development server with a mocked out model."
    echo "starting _mock_inference_api_server.py, logging output to: ${SERVER_LOG_FILE}"
    # use -u to stop buffering of output
    # use /mnt/ files to run with latest code changes on startup
    python -u /mnt/src/proxy.py > ${PROXY_LOG_FILE} 2>&1 &
    python -u /mnt/src/_mock_inference_api_server.py > ${SERVER_LOG_FILE} 2>&1 &
else
    # use -u to stop buffering of output
    python -u src/proxy.py > ${PROXY_LOG_FILE} 2>&1 &
    echo "starting inference_api_server.py, logging output to: ${SERVER_LOG_FILE}"
    gunicorn --config src/gunicorn.conf.py > ${SERVER_LOG_FILE} 2>&1 &
fi
# if this script
echo "Start up complete. Script is sleeping to keep docker container up."
sleep infinity
