#!/bin/bash
set -eu

test -n "${MODEL_WEKA_DIR}" || (echo "MODEL_WEKA_DIR must be set" && false)
test -n "${JWT_SECRET}" || (echo "JWT_SECRET must be set" && false)

echo "starting proxy ..."
nohup python inference-api/proxy.py > logfile_proxy.log 2>&1 &

echo "starting inference_api_server ..."
nohup gunicorn --config gunicorn.conf.py > logfile_inference_api_server.log 2>&1 &
