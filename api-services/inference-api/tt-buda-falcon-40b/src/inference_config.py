import os
from collections import namedtuple
from pprint import pprint


def get_env_var(var, msg):
    """Get an environment variable or raise an exception with helpful message."""
    value = os.getenv(var)
    if value is None:
        raise ValueError(f"Environment variable is required: {var}. {msg}")
    return value


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
        "backend_server_port",
        "keepalive_input_period_seconds",
        "max_seconds_healthy_no_response",
        "backend_debug_mode",
        "frontend_debug_mode",
        "mock_model",
        "falcon_config",
    ],
)

FalconConfig = namedtuple(
    "FalconConfig",
    [
        "use2layer",
        "pytorch_no_weights",
        "save",
        "load",
        "log_level_debug",
        "tti_suffix",
    ],
)

# Do as much environment variable termination here as possible.
# The exception is secrets, which are used directly as os.getenv() calls.
# get_env_var() is used to add helpful documentation for environment variables
CACHE_ROOT = get_env_var("CACHE_ROOT", msg="Base path for all data caches.")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", 7000))
MOCK_MODEL = bool(int(os.getenv("MOCK_MODEL", 0)))
BACKEND_DEBUG_MODE = bool(int(os.getenv("BACKEND_DEBUG_MODE", 0)))
FRONTEND_DEBUG_MODE = bool(int(os.getenv("FRONTEND_DEBUG_MODE", 0)))
FALCON_40B_2LAYER = bool(int(os.environ.get("FALCON_40B_2LAYER", "0")))
FALCON_40B_PYTORCH_NO_WEIGHTS = bool(
    int(os.environ.get("FALCON_40B_PYTORCH_NO_WEIGHTS", "0"))
)
FALCON_40B_SAVE = bool(int(os.getenv("FALCON_40B_SAVE", "0")))
FALCON_40B_LOAD = bool(int(os.getenv("FALCON_40B_LOAD", "0")))
FALCON_40B_LOG_LEVEL_DEBUG = bool(int(os.getenv("FALCON_40B_LOG_LEVEL_DEBUG", "0")))
FALCON_40B_TTI_SUFFIX = os.getenv("FALCON_40B_TTI_SUFFIX", "v0")

inference_config = InferenceConfig(
    hf_cache=f"{CACHE_ROOT}/hf_cache",
    tvm_cache=f"{CACHE_ROOT}/tvm_cache",
    tti_cache=f"{CACHE_ROOT}/tti_cache",
    log_cache=f"{CACHE_ROOT}/logs",
    max_input_qsize=4,  # last in queue can get response before request timeout
    input_timeout=30,  # input q backpressure, timeout in seconds
    max_inactive_seconds=60.0,  # maximum time between decode reads to be active
    backend_server_port=SERVICE_PORT,
    keepalive_input_period_seconds=120,
    max_seconds_healthy_no_response=600,
    backend_debug_mode=BACKEND_DEBUG_MODE,
    frontend_debug_mode=FRONTEND_DEBUG_MODE,
    mock_model=MOCK_MODEL,
    falcon_config=FalconConfig(
        use2layer=FALCON_40B_2LAYER,
        pytorch_no_weights=FALCON_40B_PYTORCH_NO_WEIGHTS,
        save=FALCON_40B_SAVE,
        load=FALCON_40B_LOAD,
        log_level_debug=FALCON_40B_LOG_LEVEL_DEBUG,
        tti_suffix=FALCON_40B_TTI_SUFFIX,
    ),
)

print("using inference_config:\n")
pprint(inference_config._asdict())
print("\n")
