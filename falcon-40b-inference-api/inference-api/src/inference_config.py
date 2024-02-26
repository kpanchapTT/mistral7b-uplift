import os
from collections import namedtuple

InferenceConfig = namedtuple(
    "InferenceConfig",
    [
        "hf_cache",
        "tvm_cache",
        "tti_cache",
        "log_cache",
        "max_input_qsize",
        "input_timeout",
        "max_inactive_seconds",
        "reverse_proxy_port",
        "backend_server_port",
        "keepalive_input_period_seconds",
        "max_seconds_healthy_no_response",
        "backend_debug_mode",
        "frontend_debug_mode",
    ],
)

inference_config = InferenceConfig(
    hf_cache=f"{os.environ['MODEL_WEKA_DIR']}/hf_cache",
    tvm_cache=f"{os.environ['MODEL_WEKA_DIR']}/tvm_cache",
    tti_cache=f"{os.environ['MODEL_WEKA_DIR']}/tti_cache",
    log_cache=f"{os.environ['MODEL_WEKA_DIR']}/logs",
    max_input_qsize=4,  # last in queue can get response before request timeout
    input_timeout=30,  # input q backpressure, timeout in seconds
    max_inactive_seconds=60.0,  # maximum time between decode reads to be active
    reverse_proxy_port=1223,
    backend_server_port=7000,
    keepalive_input_period_seconds=120,
    max_seconds_healthy_no_response=600,
    backend_debug_mode=bool(int(os.environ.get("BACKEND_DEBUG_MODE", 0))),
    frontend_debug_mode=bool(int(os.environ.get("FRONTEND_DEBUG_MODE", 0))),
)
