import os
from collections import namedtuple

InferenceConfig = namedtuple(
    "InferenceConfig",
    [
        "hf_cache",
        "tvm_cache",
        "tti_cache",
        "number_layers",
        "max_input_qsize",
        "input_timeout",
        "max_inactive_seconds",
        "reverse_proxy_port",
        "backend_server_port",
    ],
)

inference_config = InferenceConfig(
    hf_cache=f"/mnt/{os.environ['MODEL_WEKA_DIR']}/hf_cache",
    tvm_cache=f"/mnt/{os.environ['MODEL_WEKA_DIR']}/tvm_cache",
    tti_cache=f"/mnt/{os.environ['MODEL_WEKA_DIR']}/tti_cache",
    number_layers=1,
    max_input_qsize=64,
    input_timeout=30,  # input q backpressure, timeout in seconds
    max_inactive_seconds=60.0,  # maximum time between decode reads to be active
    reverse_proxy_port=1223,
    backend_server_port=7000,
)
