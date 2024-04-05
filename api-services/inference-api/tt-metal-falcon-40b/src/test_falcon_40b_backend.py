import queue
import os
from pathlib import Path
import time
from unittest.mock import Mock, patch, MagicMock
import sys

import torch

# Note: because falcon_40b_backend.py uses direct imports using `unittest.mock`
# does not work. The alternative would require rewritting the backend implementation

mock_tt_lib = MagicMock()
mock_fused_ops = MagicMock()
mock_conv = MagicMock()

# Assign the nested structure to the mocks
mock_tt_lib.fused_ops = mock_fused_ops
mock_fused_ops.conv = mock_conv
mock_conv.conv = MagicMock(return_value=None)  # Mock the conv function

# Mock the device part of tt_lib
mock_device = MagicMock()
mock_tt_lib.device = mock_device
mock_device.Arch = MagicMock()  # Mock the Arch class or attribute

# Insert the mocks into sys.modules
sys.modules['tt_lib'] = mock_tt_lib
sys.modules['tt_lib.fused_ops'] = mock_fused_ops
sys.modules['tt_lib.fused_ops.conv'] = mock_conv
sys.modules['tt_lib.device'] = mock_device

# mock_tt_metal_impl = MagicMock()
# mock_tt_metal_impl.TtFalconCausalLM = TtFalconCausalLM

sys.modules['tt_lib'] = mock_tt_lib
# sys.modules['tt_metal_impl.tt.falcon_causallm'] = mock_tt_metal_impl
# sys.modules['tt_metal_impl.tt.model_config'] = mock_tt_model_config
# sys.modules['tt_metal_impl.utility_functions'] = mock_tt_utility_functions

from falcon_40b_backend import run_backend, PrefillDecodeBackend
from inference_api_server import get_user_parameters
from tt_metal_impl.tt.falcon_causallm import TtFalconCausalLM

def mock_init_tt_metal(self, *args, **kwargs):
    print("using mock_init_tt_metal")
    self.num_devices = 8
    self.devices = [MagicMock() for i in range(8)]

def mock_forward(self, input_embeddings, llm_mode, attention_mask, layer_past, layer_past_len, use_cache):
    """ mock for:
    tt_logits, self.kv_cache = self.tt_FalconCausalLM(
        input_embeddings=tt_decode_embeddings,
        llm_mode="decode",
        attention_mask=tt_decode_attention_mask,
        layer_past=self.kv_cache,
        layer_past_len=self.kv_cache_len,
        use_cache=self.use_cache,
    )
    tt_logits: shape: [batch_size, vocab_size]
    """
    # mock with repeating previous token
    tps = 3000  # simulate a given tokens per second per user
    time.sleep(1 / tps)
    # output_tokens = self.input_ids[-1].unsqueeze(0)
    # # if user has hit max_length, send eos token
    # for idx, user in enumerate(self.users):
    #     if user is not None:
    #         output_tokens[0, idx] = self.tokenizer(
    #             str(user.position_id % 10)
    #         ).input_ids[0]
    #         if (user.position_id - user.prompt_length + 1) >= user.max_tokens:
    #             output_tokens[0, idx] = self.tokenizer.eos_token_id
    #         elif (
    #             (user.stop_sequence is not None)
    #             and (user.position_id - user.prompt_length + 1) > 0
    #             and (output_tokens[0, idx] == user.stop_sequence)
    #         ):
    #             output_tokens[0, idx] = self.tokenizer.eos_token_id
    # # update the new tokens generated to the input id
    # self.input_ids = output_tokens.view(1, self.max_users)
    tt_logits = torch.rand((32, self.config.vocab_size))
    return tt_logits, layer_past


@patch.object(TtFalconCausalLM, "__call__", new=mock_forward)
@patch.object(PrefillDecodeBackend, "init_tt_metal", new=mock_init_tt_metal)
def test_falcon_40b_backend():
    prompt_q = queue.Queue()
    output_q = queue.Queue()
    status_q = queue.Queue()
    # user_id, prompt, params
    default_params, _ = get_user_parameters({})
    prompt_q.put(("INIT_ID-1", "How do you get to Carnegie Hall?", default_params))
    prompt_q.put(("INIT_ID-2", "Another prompt", default_params))
    run_backend(prompt_q, output_q, status_q, verbose=False)
    print("finished")


if __name__ == "__main__":
    test_falcon_40b_backend()
