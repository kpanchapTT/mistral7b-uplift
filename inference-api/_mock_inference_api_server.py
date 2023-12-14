import os
from time import sleep
from unittest.mock import Mock, patch

from decode_backend_v1 import DecodeBackend
from inference_api_server import (app, get_backend_override_args,
                                  initialize_decode_backend)
from inference_config import inference_config

"""
This script runs the flask server and initialize_decode_backend()
with the actual model mocked out.

This allows for rapid testing of the server and backend implementation.
"""


def mock_decoder(self):
    # mock with repeating previous token
    sleep(0.1)  # 10 TPS
    output_tokens = self.input_ids[-1].unsqueeze(0)
    # if user has hit max_length, send eos token
    for idx, user in enumerate(self.users):
        if user is not None:
            output_tokens[0, idx] = self.tokenizer(
                str(user.position_id % 10)
            ).input_ids[0]
            if (user.position_id - user.prompt_length + 1) >= user.max_tokens:
                output_tokens[0, idx] = self.tokenizer.eos_token_id
            elif (
                (user.stop_sequence is not None)
                and (user.position_id - user.prompt_length + 1) > 0
                and (output_tokens[0, idx] == user.stop_sequence)
            ):
                output_tokens[0, idx] = self.tokenizer.eos_token_id
    # update the new tokens generated to the input id
    self.input_ids = output_tokens.view(1, self.max_users)


def mock_load_model_and_tokenizer(self, args):
    from transformers import AutoTokenizer

    # # mock model
    model = None
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.hf_cache)
    return model, tokenizer


def mock_post_init_pybudify(self, args):
    pass


@patch.object(DecodeBackend, "decode", new=mock_decoder)
@patch.object(DecodeBackend, "_post_init_pybudify", new=mock_post_init_pybudify)
@patch.object(
    DecodeBackend, "load_model_and_tokenizer", new=mock_load_model_and_tokenizer
)
def test_server():
    if not os.path.exists("server_logs"):
        os.makedirs("server_logs")
    override_args = get_backend_override_args()
    initialize_decode_backend(override_args)
    app.run(
        debug=True,
        port=inference_config.backend_server_port,
        host="0.0.0.0",
        use_reloader=False,
    )


if __name__ == "__main__":
    test_server()