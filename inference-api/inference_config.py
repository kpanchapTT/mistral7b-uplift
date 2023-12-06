import os
from collections import namedtuple

InferenceConfig = namedtuple(
    "InferenceConfig",
    [
        "hf_cache",
        "number_layers",
        "max_input_qsize",
        "input_timeout",
        "max_inactive_seconds",
        "reverse_proxy_port",
        "backend_server_port",
    ],
)

inference_config = InferenceConfig(
    hf_cache="/proj_sw/large-model-cache/falcon40b",
    number_layers=1,
    max_input_qsize=64,
    input_timeout=30,  # input q backpressure, timeout in seconds
    max_inactive_seconds=60.0,  # maximum time between decode reads to be active
    reverse_proxy_port=1223,
    backend_server_port=7000,
)
