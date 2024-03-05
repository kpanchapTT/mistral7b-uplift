# Galaxy Falcon 40B Inference API Deployment

1. build and push new docker image
    - manual because of side build of pybuda and BBE
2. tt-cloud shutdown instance (because there is currently only 1 galaxy in cloud k8s)
    - https://cloud.tenstorrent.com/tenstorrent-playground
    - remove any static ingress URLs allocated if existing
3. tt-cloud run new workload
    - ensure correct settings
        - Environment Variables:
            - JWT_SECRET=<SECRET>
            - CACHE_ROOT=/mnt/falcon-galaxy-store
            - FALCON_40B_LOAD=1
            - FALCON_40B_TTI_SUFFIX=v4_k8s
        - Memory: 492Gi
        - Persistent storage mounts:
            - falcon-galaxy-store 512GB
    - set Image: e.g. `ghcr.io/tenstorrent/project-falcon:v0.0.17`
    - Runtime: e.g. Wormhole Galaxy (kmd 1.26, firmware 2023-11-07)
    - click "start workload"
4. check status using tt-cli, until `STATUS=Running`
    - potential issues:
        - new image may take greater than 5 minutes timeout to download, this can put the pod into a bad state and stop the galaxy host from being freed back into k8s resource pool. Requires admin panel to fix, ask cloud team for support.
        - requested too much memory in container, cannot create. Delete workload and set lower memory.
        - 
5. log in to k8s pod using tt-cli (see instructions below) -- this is only needed to compile the model or view logs
6. reset k8s galaxy (see instructions below)
7. start tmux
    - tmux is needed because we may want to log back in to check state or reset galaxy
8. [ONLY IF NEEDED] compile model (see instructions below)
12. assign ingress route: https://falcon-api--tenstorrent-playground.workload.tenstorrent.com/
    - potential issues:
        - workload may be stuck in `pending` state on GUI, this hides the ingress route option on the workload. Requires admin panel to fix, ask cloud team for support.
13. Test API from remote:
```bash
./scripts/jwt_util.py --secret ${JWT_SECRET} encode '{"team_id": "tenstorrent", "token_id":"debug-test"}'
<SECRET>
export AUTHORIZATION="Bearer <SECRET>"
export LLM_CHAT_API_URL=https://falcon-api--tenstorrent-playground.workload.tenstorrent.com/predictions/falcon40b
curl ${LLM_CHAT_API_URL} -L -H "Content-Type: application/json" \
-H "Authorization: ${AUTHORIZATION}" \
-d '{"text":"Write a short play with starring Janet", "top_p":"0.9", "top_k": "40", "temperature": "1.2", "max_tokens": "128"}'
```

## log in to k8s pod
see: https://github.com/tenstorrent/cloud/tree/main/cli
```bash
# node and pod name for example only
export REMOTE_NODE=falcon-40b-galaxy-api-39aaaf2a
export REMOTE_POD=falcon-40b-galaxy-api-39aaaf2a-deployment-5b5b6848cd-zzlbj

ttcloud pods -n "${REMOTE_NODE}"
ttcloud exec -n "${REMOTE_NODE}" "${REMOTE_POD}"
```

## reset k8s galaxy
NOTE: ensure correct galaxy IP address before running, if unsure get IP address from cloud team.
```bash
# run on local, cannot run from within colo, but needs to be on colo VPN
date && curl ${GALAXY_IP}:8000/shutdown/modules -u admin:admin -X POST -s '{"groups": null}'
# run on pod:
# /usr/local/bin/tt-smi -lr 0
date && ./tt-smi-wh-8.C.0.0_2023-11-02-ddcfb4b7bb67635e -lr 0
# before it comes back up run on local
date && curl ${GALAXY_IP}:8000/boot/modules -u admin:admin -X POST -d '{"groups": null}'
# check 32 modules + n150 are found and ethernet links have trained
date && ./tt-smi-wh-8.C.0.0_2023-11-02-ddcfb4b7bb67635e
```

## compile model

The decode.py script is used because some environment variables are set to different defaults within the inference_api_server.py script.

2L model for testing debugging
```bash
TT_BACKEND_COMPILE_THREADS=16 PYBUDA_RELOAD_GENERATED_MODULES=1 TT_BACKEND_DRAM_POLLING_FREQUENCY=64 TT_BACKEND_PROFILER=1 PYBUDA_DEVICE_EMBEDDINGS=1 PYBUDA_MICROBATCH_LOOPING=1 python inference-api/decode_v0.py --mode sequential -l 2 --version efficient-40b -d silicon --arch nebula-galaxy --num-tokens 500 --user-rows 32 --precision bf16 --num-chips 32 -mf 8 --log-level DEBUG --opt-level 4 --hf-cache /mnt/falcon-galaxy-store/hf_cache/ --enable-tvm-cache --load-pretrained -odlmh -plmh -fv --top-k 64 --top-p 0.9 --flash-decode --save /mnt/falcon-galaxy-store/tti_cache/flash_decode_2l_v5_k8s_test.tti --model tiiuae/falcon-40b-instruct
```