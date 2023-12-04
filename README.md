# Project Falcon

Project Falcon planning documentation: https://docs.google.com/document/d/1811CseBXgP2e7wTCN0xfTa4oDKBr0ossTeuUJSffLV4/edit#heading=h.mhop4lldk8kn

# Inference API Workstream

## Phase 0: Basic non-chat

Deploy Falcon 40B in the cloud, API structure with key that you can query.

The Falcon 40B implementation is from the `large-lm` repo`: https://yyz-gitlab.local.tenstorrent.com/tenstorrent/large-lm/-/tree/1c6c521181b93612a79d8e9995acef2e45f97f02/investigations

# [WIP] Setup

Demo Repro:

Pybuda: 0cf650fbacff0f05297483bdac1fe92d7fd639f2
BBE: bcb342b9bfed881733d43826d7e739b61a126840


Run the decode server
```bash
source env/bin/activate
python inference-api/decode_server.py
```


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

export HF_CACHE="/proj_sw/large-model-cache/falcon40b"
```

# [WIP] Documentation 
The decode server calls `decode_backend_v0.py` run_decode_backend()

DecodeBackend.run_generate() runs in a loop


# Server batching

Flask SocketIO app

identify caller with flask.session.get('session_id')

# clean tt artifacts

```bash
rm -rf .hlkc_cache/ .pkl_memoize_py3/ generated_modules/ tt_build/
```