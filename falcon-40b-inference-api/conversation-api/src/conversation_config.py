import os
from collections import namedtuple

ConversationConfig = namedtuple(
    "ConversationConfig",
    [
        "log_cache",
        "reverse_proxy_port",
        "backend_server_port",
        "debug_mode",
    ],
)

conversation_config = ConversationConfig(
    log_cache=f"/logs",
    reverse_proxy_port=1223,
    backend_server_port=7000,
    debug_mode=bool(int(os.environ.get("DEBUG_MODE", 0))),
)
