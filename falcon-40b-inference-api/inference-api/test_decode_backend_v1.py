import queue
from pathlib import Path
from time import sleep
from unittest.mock import Mock, patch

import torch
from decode_backend_v1 import DecodeBackend, run_decode_backend
from inference_api_server import get_user_parameters

test_prompts_outputs = [
    ("This is a test prompt.", "this is test output, much longer now"),
    ("Another prompt.", "also test output"),
]


class MockModel:
    vocab_size = 10


def mock_decoder(self):
    # mock with repeating previous token
    sleep(0.1)  # 10 TPS
    output_tokens = torch.zeros((self.max_users), dtype=torch.long)
    # if user has hit max_length, send eos token
    for idx, user in enumerate(self.users):
        if user is not None:
            if (user.position_id + 1 - user.prompt_length) >= user.max_tokens:
                output_tokens[idx] = self.tokenizer.eos_token_id
            elif (user.position_id + 1 - user.prompt_length) >= 0:
                # done prefill, send output tokens
                out_idx = user.position_id - user.prompt_length
                out_tokens = self.tokenizer(test_prompts_outputs[idx][1]).input_ids
                if len(out_tokens) <= out_idx:
                    output_tokens[idx] = self.tokenizer.eos_token_id
                else:
                    output_tokens[idx] = out_tokens[out_idx]
                if (user.stop_sequence is not None) and (
                    output_tokens[idx] == user.stop_sequence
                ):
                    output_tokens[idx] = self.tokenizer.eos_token_id
            else:
                output_tokens[idx] = user.prompt_tokens.squeeze(0)[user.position_id + 1]
            print(
                f"mock_decoder: idx={idx}: {self.tokenizer.decode(output_tokens[idx])}"
            )

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
def test_decode_backend_v1():
    prompt_q = queue.Queue()
    output_q = queue.Queue()
    status_q = queue.Queue()

    # user_id, prompt, params
    params, _ = get_user_parameters({})
    for idx, tp in enumerate(test_prompts_outputs):
        prompt_q.put((f"test_user_id_{idx}", tp[0], params))

    arg_overrides = [
        "--mode",
        "concurrent",
        "-l",
        "60",
        "--version",
        "efficient-40b",
        "-d",
        "silicon",
        "--arch",
        "nebula-galaxy",
        "--num-tokens",
        "1000000",
        "--num-outer-loops",
        "1000",
        "--user-rows",
        "32",
        "--precision",
        "bf16",
        "--num-chips",
        "32",
        "-mf",
        "8",
        "--log-level",
        "ERROR",
        "--opt-level",
        "4",
        "--hf-cache",
        "/proj_sw/large-model-cache/falcon40b/hf_cache",
        "--enable-tvm-cache",
        "-odlmh",
        "-plmh",
        "-fv",
        "--flash-decode",
        "--top-k",
        "5",
        "--top-p",
        "0.9",
        "--load-pretrained",
        "--model",
        "tiiuae/falcon-40b-instruct",
    ]
    verbose = False
    run_decode_backend(prompt_q, output_q, status_q, arg_overrides, verbose=False)


if __name__ == "__main__":
    test_decode_backend_v1()
