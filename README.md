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
```python
python decode_server.py
```

# [WIP] Documentation 
The decode server calls `decode_backend_v0.py` run_decode_backend()

DecodeBackend.run_generate() runs in a loop


# Server batching

Flask SocketIO app

identify caller with flask.session.get('session_id')


