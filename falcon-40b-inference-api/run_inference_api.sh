#!/bin/bash
set -eu

test -n "${MODEL_WEKA_DIR}" || (echo "MODEL_WEKA_DIR must be set" && false)
test -n "${JWT_SECRET}" || (echo "JWT_SECRET must be set" && false)

# Get current date and time
START_DATETIME=$(date '+%Y-%m-%d-%H_%M_%S')
# Define the log file location
SERVER_LOG_FILE="${MODEL_WEKA_DIR}/logs/inference_api_server_${START_DATETIME}.log"
PROXY_LOG_FILE="${MODEL_WEKA_DIR}/logs/proxy_${START_DATETIME}.log"

echo "starting proxy, logging output to: ${PROXY_LOG_FILE}"
python inference-api/proxy.py > ${PROXY_LOG_FILE} 2>&1 &

echo "starting inference_api_server, logging output to: ${SERVER_LOG_FILE}"
gunicorn --config gunicorn.conf.py > ${SERVER_LOG_FILE} 2>&1 &
