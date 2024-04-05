import os
from time import sleep
from unittest.mock import Mock, patch


from test_falcon_40b_backend import mock_init_tt_metal, mock_forward

from falcon_40b_backend import PrefillDecodeBackend
from tt_metal_impl.tt.falcon_causallm import TtFalconCausalLM
from inference_api_server import (
    app,
    initialize_decode_backend,
    global_backend_init
)
from inference_config import inference_config

"""
This script runs the flask server and initialize_decode_backend()
with the actual model mocked out.

This allows for rapid testing of the server and backend implementation.
"""

backend_initialized = False
api_log_dir = os.path.join(inference_config.log_cache, "api_logs")


# def global_backend_init():
#     global backend_initialized
#     if not backend_initialized:
#         # Create server log directory
#         if not os.path.exists(api_log_dir):
#             os.makedirs(api_log_dir)
#         initialize_decode_backend()
#         backend_initialized = True


@patch.object(TtFalconCausalLM, "__call__", new=mock_forward)
@patch.object(PrefillDecodeBackend, "init_tt_metal", new=mock_init_tt_metal)
def create_test_server():
    # from flask_cors import CORS
    # # CORS for swagger-ui local testing
    # CORS(
    #     app,
    #     supports_credentials=True,
    #     resources={r"/predictions/*": {"origins": "http://localhost:8080"}},
    # )
    global_backend_init()
    return app


if __name__ == "__main__":
    app = create_test_server()
    app.run(
        port=inference_config.backend_server_port,
        host="0.0.0.0",
    )
