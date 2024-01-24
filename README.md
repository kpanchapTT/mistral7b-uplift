# Project Falcon

Project Falcon planning documentation: https://docs.google.com/document/d/1811CseBXgP2e7wTCN0xfTa4oDKBr0ossTeuUJSffLV4/edit#heading=h.mhop4lldk8kn

# Inference API Workstream

## Phase 0: Basic non-chat

Deploy Falcon 40B in the cloud, API can be accessesed using JWT, basic user parameters exposed: temperature, top_k, top_p, stop_sequence, max_tokens.

Technical documentation: [falcon-40b-inference-api/README.md](falcon-40b-inference-api/README.md)

use cases:
1. Falcon 40B integrated with AI playground
2. Accessible by customers from direct API calls using URL+JWT

The Falcon 40B implementation is from the `large-lm` repo`: https://yyz-gitlab.local.tenstorrent.com/tenstorrent/large-lm/-/tree/1c6c521181b93612a79d8e9995acef2e45f97f02/investigations

## Phase 1: Basic chat

Design in progress