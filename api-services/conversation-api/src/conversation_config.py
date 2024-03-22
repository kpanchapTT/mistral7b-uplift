import os
from collections import namedtuple


def get_env_var(var, msg):
    """Get an environment variable or raise an exception with helpful message."""
    value = os.getenv(var)
    if value is None:
        raise ValueError(f"Environment variable is required: {var}. {msg}")
    return value


ConversationConfig = namedtuple(
    "ConversationConfig",
    [
        "log_cache",
        "backend_server_port",
        "test_server",
        "debug_mode",
        "tt_falcon_40b_inference_api_url",
    ],
)

# Do as much environment variable termination here as possible.
# The exception is secrets, which are used directly as os.getenv() calls.
# get_env_var() is used to add helpful documentation for environment variables
CACHE_ROOT = get_env_var("CACHE_ROOT", msg="Base path for all data caches.")
TT_FALCON_40B_INFERENCE_API_URL = get_env_var(
    "TT_FALCON_40B_INFERENCE_API_URL", msg="URL for inference API."
)
SERVICE_PORT = int(os.getenv("SERVICE_PORT", 7000))
# defines if CORS is enabled or not
TEST_SERVER = bool(int(os.getenv("TEST_SERVER", 0)))
BACKEND_DEBUG_MODE = bool(int(os.getenv("BACKEND_DEBUG_MODE", 0)))

conversation_config = ConversationConfig(
    log_cache=f"{CACHE_ROOT}/logs",
    backend_server_port=SERVICE_PORT,
    test_server=TEST_SERVER,
    debug_mode=BACKEND_DEBUG_MODE,
    tt_falcon_40b_inference_api_url=TT_FALCON_40B_INFERENCE_API_URL,
)
