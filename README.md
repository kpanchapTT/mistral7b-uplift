# Project Falcon

Project Falcon planning documentation: https://docs.google.com/document/d/1811CseBXgP2e7wTCN0xfTa4oDKBr0ossTeuUJSffLV4/edit#heading=h.mhop4lldk8kn

# Inference API Workstream

## Phase 0: Basic non-chat

Deploy Falcon 40B in the cloud, API structure with key that you can query.

The Falcon 40B implementation is from the `large-lm` repo`: https://yyz-gitlab.local.tenstorrent.com/tenstorrent/large-lm/-/tree/1c6c521181b93612a79d8e9995acef2e45f97f02/investigations

# Setup

## env setup

```bash
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip=20.0.2 setuptools wheel
python3 -m pip install pybuda-0.1.231113+dev.wh.b0.cdbd30a-cp38-cp38-linux_x86_64.whl
python3 -m pip install tvm-0.9.0+dev.tt.c2076affc-cp38-cp38-linux_x86_64.whl
python3 -m pip install -r requirements.txt
# for developer tools
python3 -m pip install -r requirements_dev.txt

# will be different on other machines
export MODEL_WEKA_DIR=/proj_sw/large-model-cache/falcon40b
# export HF_CACHE="/proj_sw/large-model-cache/falcon40b/hf_cache"
```

## Inference API server

The Inference API server is a Flask web server using HTTP 1.1 chunked encoding for streaming responses token by token.

Run the inference API server:
```bash
source env/bin/activate
python inference-api/inference_api_server.py
```

## proxy

The proxy handles JWT authentication by request header checking and is a reverse proxy to incoming traffic.

```bash
source env/bin/activate
export JWT_SECRET=test-secret-123
python proxy/proxy.py
```

### JWT Authorization

To authenticate requests use the header `Authorization`. The JWT token can be computed using the script `jwt_util.py`. This is an example:
```bash
./scripts/jwt_util.py --secret ${JWT_SECRET} encode '{"team_id": "tenstorrent", "token_id":"debug-test"}'
export AUTHORIZATION="Bearer <token>"
```

Warning: do not use weak secrets like the example one in anything public facing!

# Run tests

```bash
source env/bin/activate
python inference-api/test_decoding.py
# this requires a running inference server as described above
python inference-api/test_inference_api.py
```

For testing of just the flask server a mock_backend is available:
```bash
source env/bin/activate
export MODEL_WEKA_DIR=/proj_sw/large-model-cache/falcon40b
python inference-api/_mock_server_.py
# run in separate terminals or processes
python inference-api/test_inference_api.py
```

# Docker images

## Building Docker images

If running on a macbook or non x86_64 platform use the `--platform flag`:
```bash
docker build -t project-falcon/falcon40b-demo:1.0.0 --platform=linux/x86_64 .
```

## Running docker images

By default Docker containers have access to all of the host's CPU cores and RAM.

Here is an example mapping the resources to the docker container from a BM host connected to a galaxy, this is a similar configuration to how cloud k8s variables would be set:

```bash
docker run --rm -ti \
    --shm-size=4g \
    --device /dev/tenstorrent \
    -p 1223:1223 \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    -v /proj_sw/large-model-cache/falcon40b:/mnt/falcon-galaxy-store/hf_cache \
    -v /proj_sw/large-model-cache/falcon40b/tvm_cache:/mnt/falcon-galaxy-store/tvm_cache \
    -v /home/tt-admin/project-falcon/:/mnt/falcon-galaxy-store/tti_cache \
    -e MODEL_WEKA_DIR=/mnt/falcon-galaxy-store \
    -e JWT_SECRET=test-secret-456\
    project-falcon/falcon40b-demo:0.0.0-test bash
```

For testing the following environment variables can be used to switch the backend server override arguments:
```bash
-e FALCON_40B_2LAYER='1' 
-e FALCON_40B_SAVE='1'
-e FALCON_40B_LOAD='1'
-e FALCON_40B_PYTORCH_NO_WEIGHTS='1' 
```

# [WIP] Documentation 

`inference_api_server.py` runs the main flask server that is behind the proxy.
This file starts `decode_backend_v1.run_decode_backend` as a `multiprocessing.Process`

`DecodeBackend.run_generate` is the main backend, it connects with the flask server using 3 `multiprocessing.Queue`s:
- input_queue
- output_queue
- status_queue

## Life cycle of requests

Each new request is handled on a separate thread. The prompt is validated then pushed onto the input_queue.

The backend aysnchronously processes input_queue prompts into available user_rows in the decoding. Currently a fixed 32 user_rows is supported. The prefill of the on device KV cache (ODKV) is done by the same graph on the TT device.

Each token is pushed onto the output_queue once it is generated. The output_queue is read aysnchronously by a single threaded process within the flask server and stored in a dict of queues keyed on the session_id (or user_id), this means no two users can get eachother's response tokens. Each token is read by the request processes from this dict using the corresponding session_id and streamed back to the user using HTTP 1.1 chunked encoding.

When decoding is finished for a user within the user_rows it is evicted from the `DecodeBackend.users` list. On the next loop over inputing data from the input_queue this user_row will be filled if there is input prompt data for it.

The flask server request process having received the EOS token marks the connection closed and it's session_id as ready for deletion in the dict of session_id keyed queues.

# clean tt artifacts

```bash
rm -rf .hlkc_cache/ .pkl_memoize_py3/ generated_modules/ tt_build/
```