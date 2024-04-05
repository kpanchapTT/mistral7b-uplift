import os
import time
import traceback
from multiprocessing import Queue
from pathlib import Path

# import json
# import pytest
from functools import partial
# import tt_lib
import torch
# from loguru import logger
# import time
from transformers import AutoTokenizer
# import os

from transformers.generation.utils import top_k_top_p_filtering
from inference_config import inference_config
from inference_logger import get_logger
import torch.nn.functional as F

import tt_lib
from tt_metal_impl.reference.hf_modeling_falcon import FalconConfig, FalconForCausalLM
from tt_metal_impl.tt.falcon_common import PytorchFalconCausalLM
from tt_metal_impl.tt.falcon_causallm import TtFalconCausalLM
from tt_metal_impl.tt.model_config import get_model_config, model_config_entries
from tt_metal_impl.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    profiler,
    torch2tt_tensor,
    tt2torch_tensor,
    nearest_32,
    get_devices_for_t3000,
)


logger = get_logger(__name__)
logger.info(f"importing {__name__}")
END_OF_TEXT = 11
SPACE = 204


def post_process(logits, index):
    next_token_logits = logits[:, index, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    ids = next_tokens[:, None]
    return ids


def preprocess_and_validate_inputs(input_prompts, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token

    num_users = len(input_prompts)
    num_input_tokens = -1
    for prompt in input_prompts:
        tokenized_prompt = tokenizer(prompt, padding=False, add_special_tokens=False, return_tensors="pt")
        num_input_tokens = max(num_input_tokens, len(tokenized_prompt["input_ids"][0]))

    seq_len_padded_to_32 = nearest_32(num_input_tokens)
    assert seq_len_padded_to_32 == 32, "Prefill only supports 32 tokens max"

    tokenized_inputs = tokenizer(
        input_prompts,
        padding="max_length",
        max_length=seq_len_padded_to_32,
        add_special_tokens=False,
        return_tensors="pt",
    )
    prefill_ids = tokenized_inputs["input_ids"]

    logger.info(f"# of users: {num_users}")
    logger.info(f"# of input tokens per user: {num_input_tokens}")

    return prefill_ids, num_users, num_input_tokens


def initialize_kv_cache(model_config, configuration, num_layers, batch_size, max_seq_len, devices):
    logger.info("Filling kv cache on device")
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    num_kv_heads = configuration.num_kv_heads

    tt_kv_cache = ()
    tt_k_cache_host = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim)
    tt_v_cache_host = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim)
    tt_k_cache_host = torch.chunk(tt_k_cache_host, len(devices), 1)
    tt_v_cache_host = torch.chunk(tt_v_cache_host, len(devices), 1)

    for i in range(num_layers):
        logger.info(f"Initializing kv cache on devices for layer: {i+1}")
        tt_k_cache = []
        tt_v_cache = []
        for j in range(len(devices)):
            tt_k_cache.append(
                torch2tt_tensor(
                    tt_k_cache_host[j],
                    devices[j],
                    tt_lib.tensor.Layout.TILE,
                    model_config["KV_CACHE_MEMCFG"],
                    model_config["KV_CACHE_DTYPE"],
                )
            )
            tt_v_cache.append(
                torch2tt_tensor(
                    tt_v_cache_host[j],
                    devices[j],
                    tt_lib.tensor.Layout.TILE,
                    model_config["KV_CACHE_MEMCFG"],
                    model_config["KV_CACHE_DTYPE"],
                )
            )
        tt_kv_cache += ((tt_k_cache, tt_v_cache),)
    return tt_kv_cache


# TODO: Remove once we have prefill on device
def initialize_and_fill_kv_cache(
    pytorch_FalconCausalLM, model_config, configuration, prefill_ids, num_layers, batch_size, max_seq_len, devices
):
    logger.info("Generating kv cache on host")

    pytorch_out, pytorch_layer_present = pytorch_FalconCausalLM(
        input_ids=prefill_ids, past_key_values=None, use_cache=True
    )

    head_dim = configuration.hidden_size // configuration.num_attention_heads
    q_heads_per_kv_heads = configuration.num_attention_heads // configuration.num_kv_heads
    num_users, kv_cache_len = prefill_ids.shape

    # TODO: Remove this debug code; uncomment to use dummy cache
    # pytorch_out = torch.rand(num_users, kv_cache_len, 65024)
    # single_layer_cache = (torch.rand(num_users, 128, kv_cache_len, 64), torch.rand(num_users, 128, kv_cache_len, 64))
    # pytorch_layer_present = (single_layer_cache,) * 60

    kv_cache = ()
    for i in range(num_layers):
        logger.info(f"Putting kv cache on devices for layer: {i+1}")
        k_cache_repeat_interleaved, v_cache_repeat_interleaved = pytorch_layer_present[i]
        k_cache = k_cache_repeat_interleaved[:, ::q_heads_per_kv_heads, ...]
        v_cache = v_cache_repeat_interleaved[:, ::q_heads_per_kv_heads, ...]

        tt_k_cache_host = torch.zeros(batch_size, configuration.num_kv_heads, max_seq_len, head_dim)
        tt_v_cache_host = torch.zeros(batch_size, configuration.num_kv_heads, max_seq_len, head_dim)
        tt_k_cache_host[:num_users, :, :kv_cache_len, :] = k_cache
        tt_v_cache_host[:num_users, :, :kv_cache_len, :] = v_cache
        tt_k_cache_host = torch.chunk(tt_k_cache_host, len(devices), 1)
        tt_v_cache_host = torch.chunk(tt_v_cache_host, len(devices), 1)

        tt_k_cache = []
        tt_v_cache = []
        for j in range(len(devices)):
            tt_k_cache.append(
                torch2tt_tensor(
                    tt_k_cache_host[j],
                    devices[j],
                    tt_lib.tensor.Layout.TILE,
                    model_config["KV_CACHE_MEMCFG"],
                    model_config["KV_CACHE_DTYPE"],
                )
            )
            tt_v_cache.append(
                torch2tt_tensor(
                    tt_v_cache_host[j],
                    devices[j],
                    tt_lib.tensor.Layout.TILE,
                    model_config["KV_CACHE_MEMCFG"],
                    model_config["KV_CACHE_DTYPE"],
                )
            )
        kv_cache += ((tt_k_cache, tt_v_cache),)

    return pytorch_out, kv_cache


def print_output_prompts(generated_ids, tokenizer, num_users_to_display=None):
    output_prompts = tokenizer.batch_decode(generated_ids.tolist())
    for user_id, output_prompt in enumerate(output_prompts[:num_users_to_display]):
        logger.info(f"Output for user {user_id}:\n{output_prompt}")



class UserInfo:
    def __init__(self, user_id, prompt, position_id, params, tokenizer):
        self.user_id = user_id
        self.prompt = prompt
        self.position_id = position_id
        # TODO: only tokenize once, consolidate with preprocess_and_validate_inputs()
        tokenized = tokenizer(
            prompt,
            padding="max_length",
            return_tensors="pt",
            max_length=2048,
            truncation=True,
        )
        # remove any EOS tokens for input
        tokenized.input_ids = tokenized.input_ids[
            tokenized.input_ids != tokenizer.eos_token_id
        ]
        # pad back to 2048 tokens
        tokenized.input_ids = F.pad(
            tokenized.input_ids,
            (0, 2048 - tokenized.input_ids.size(0)),
            "constant",
            0,
        )

        self.prompt_tokens = tokenized.input_ids.clone().squeeze()  # (2048,)
        self.prompt_length = torch.sum(tokenized.attention_mask).item()  # int
        self.num_tokens_generated = 0
        self.stop_sequence = None
        self.generation_params = params
        self.max_tokens = params["max_tokens"]
        self.return_prompt = params["return_prompt"]
        self.cancel = False
        self.prefill_complete = False
        self.decode_complete = False
        self.sent_stop = False

        if params.get("stop_sequence"):
            self.stop_sequence = tokenizer(params.get("stop_sequence")).input_ids[0]


class PrefillDecodeBackend:
    def __init__(
        self,
        model_version,
        batch_size,
        num_layers,
        max_seq_len,
        cache_root,
        verbose=False,
    ) -> None:
        """
        Initialize pybuda model and all infracstructures to continuously run decode
        Maintain a cur_prompts for decode.
        """
        self.max_users = 32
        self.num_users = None
        self.users = [None for _ in range(self.max_users)]
        self.use_cache = True
        # # inputs to model
        self.decode_ids = None
        # backend status
        self.time_last_status = time.time()
        self.update_period = 1  # status message period in seconds
        self.num_steps = 0
        self.verbose = verbose  # enable conditional debug logging
        # new init:
        self.model_version = model_version
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        #
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_version)
        self.tokenizer.pad_token_id = 0
        self.post_processor = partial(post_process)
        self.default_top_p = inference_config.falcon_config.default_top_p
        self.default_top_k = inference_config.falcon_config.default_top_k
        self.default_temperature = inference_config.falcon_config.default_temperature
        #
        self.timestamps_start = {}
        self.timestamps_stop = {}
        self.enable_profile_logging = False
        #
        self.cache_root = Path(cache_root)
        if not self.cache_root.exists():
            self.cache_root.mkdir(parents=True, exist_ok=True)
        # initialization
        self.prefill_on_host = True
        self.use_cache = True
        self.devices = []
        self.configuration = None
        self.model_config = None
        self.init_tt_metal()
        self.init_model()

    def get_users(self):
        return [u for u in self.users if u]

    def get_user_param(self, param):
        return [
            user.generation_params[param] if user is not None else None
            for user in self.users
        ]

    def timer_start(self, name):
        self.timestamps_start[name] = time.time()

    def timer_stop(self, name, log=False):
        if name in self.timestamps_start.keys():
            self.timestamps_stop[name] = time.time()
            timedelta = self.timestamps_stop[name] - self.timestamps_start[name]
            if log or self.enable_profile_logging:
                print(f"timedelta: {name}: {timedelta} seconds")
                logger.info(f"timedelta: {name}: {timedelta} seconds")

    def model_location_generator(self, model_version, model_subdir=""):
        model_cache_path = Path(self.cache_root) / "tt-metal-models" / model_version
        model_cache_path.mkdir(parents=True, exist_ok=True)
        return model_cache_path

    def get_tt_cache_path(self, model_version, model_subdir="", default_dir=""):
        tt_cache_path = Path(self.cache_root) / "tt-metal-cache" / model_version
        tt_cache_path.mkdir(parents=True, exist_ok=True)
        return tt_cache_path

    def teardown(self):
        logger.info("teardown ...")
        if not os.environ.get("MOCK_MODEL"):
            self.teardown_tt_metal_device()

    def teardown_tt_metal_device(self):
        logger.info("teardown_tt_metal_device ...")
        # ttl.device.CloseDevice(self.device)
        # ttl.device.DeallocateBuffers(self.device)
        # ttl.program_cache.disable_and_clear()
        for device in self.devices:
            device.disable_and_clear_program_cache()
        # from: use_program_cache
        # from: all_devices
        for device in self.devices.values():
            tt_lib.device.DumpDeviceProfiler(device, True)
            tt_lib.device.DeallocateBuffers(device)

        tt_lib.device.CloseDevices(self.devices)
        

    def init_tt_metal_device(self):
        logger.info("init_tt_metal_device ...")
        # from: all_devices
        num_devices = tt_lib.device.GetNumAvailableDevices()

        # Get only physical devices
        devices = tt_lib.device.CreateDevices([i for i in range(num_devices)])

        all_devices = [devices[i] for i in range(num_devices)]
        
        self.devices = get_devices_for_t3000(all_devices, num_devices)

        # from: use_program_cache
        for dev in self.devices:
            dev.enable_program_cache()



    def init_tt_metal(self):
        logger.info("init_tt_metal ...")
        self.init_tt_metal_device()
        # from: test_demo()
        # disable_persistent_kernel_cache()
        disable_compilation_reports()

    def init_model(self):
        # Set it up for prefill initially and change the model_config to decode
        model_config_str = "BFLOAT8_B-SHARDED"
        _init_model_config = get_model_config(model_config_str, "prefill", [1, 32], self.num_devices)
        # model_version = model_config_entries["_name_or_path"]
        self.tt_cache_path = self.get_tt_cache_path(self.model_version)
        
        # from: run_falcon_demo_kv(
        #     user_input=user_input,
        #     model_version=model_version,
        #     model_config_str=model_config_str,
        #     model_config=model_config,
        #     batch_size=32,
        #     num_layers=model_config_entries["num_hidden_layers"],
        #     max_seq_len=128,  # 1024,
        #     model_location_generator=model_location_generator,
        #     tt_cache_path=tt_cache_path,
        #     devices=devices,
        #     prefill_on_host=True,
        # )
        torch.manual_seed(0)
        # TODO: remove as redundant? -> use_program_cache
        # for device in self.devices:
        #     device.enable_program_cache()

        self.configuration = FalconConfig(**model_config_entries)

        # State dict is needed for embeddings
        logger.info("Loading weights...")
        profiler.start(f"loading_weights")
        if len(os.listdir(self.tt_cache_path)) < 260:
            logger.info("Weights not found on machine; downloading weights...")
            model_cache = self.model_location_generator(self.model_version)
            # use cache_dir arg
            hugging_face_reference_model = FalconForCausalLM.from_pretrained(
                self.model_version, low_cpu_mem_usage=True, cache_dir=model_cache
            )
            hugging_face_reference_model.eval()
            state_dict = hugging_face_reference_model.state_dict()
            torch.save(
                state_dict["transformer.word_embeddings.weight"],
                self.tt_cache_path / "embedding.pt",
            )
        else:
            state_dict = None

        logger.info("Loading weights finished!")
        profiler.end(f"loading_weights")

        for device in self.devices:
            tt_lib.device.Synchronize(device)

        logger.info("Moving weights to device; might take some time...")
        profiler.start(f"moving_to_device")

        base_url = ""
        use_global_cos_sin_cache = True
        self.tt_FalconCausalLM = TtFalconCausalLM(
            self.device,
            state_dict,
            base_url,
            self.num_layers,
            self.configuration,
            self.max_seq_len,
            _init_model_config,
            self.tt_cache_path,
            use_global_cos_sin_cache,
        )

        logger.info("Moved weights to device!")
        if self.prefill_on_host:
            # TODO: Remove pytorch model once prefill is on device
            logger.info("Loading PyTorch model for prefill")
            hugging_face_reference_model = FalconForCausalLM.from_pretrained(self.model_version, low_cpu_mem_usage=True)
            hugging_face_reference_model.eval()
            self.pytorch_FalconCausalLM = PytorchFalconCausalLM(hugging_face_reference_model, self.num_layers)
        else:
            logger.info("Initializing and KV cache")
            profiler.start(f"initializing_KV_cache")
            kv_cache = initialize_kv_cache(_init_model_config, self.configuration, self.num_layers, self.batch_size, self.max_seq_len, self.devices)
            profiler.end(f"initializing_KV_cache")


        logger.info("initializing decode ...")
        # Update model_config for decode
        # TODO: when adding support for prefill on device this should change
        # likely cannot be in init
        self.model_config = get_model_config(model_config_str, "decode", [self.batch_size, 1], len(self.devices))
        self.tt_FalconCausalLM.set_model_config(self.model_config)
        self.attention_mask_memconfig = self.model_config["ATTN_MASK_MEMCFG"]
        if self.attention_mask_memconfig.is_sharded():
            attn_mask_shard_shape = self.attention_mask_memconfig.shard_spec.shape
            attn_mask_shard_shape[-1] = self.max_seq_len
            self.attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape

        logger.info("init_tt_metal completed.")

    def _get_user_by_id(self, user_id):
        for user in self.users:
            if user is not None and user.user_id == user_id:
                return user
        return None

    def _get_num_of_users(self):
        # find num of non None users
        return sum([1 for user in self.users if user is not None])

    def _find_free_user_slot(self):
        """return the index of the first free user slot"""
        for i, user in enumerate(self.users):
            if user is None:
                return i

    def _add_users_from_non_empty_queue(self, prompt_q):
        """add users from prompt_q to self.users"""
        while not prompt_q.empty() and self._get_num_of_users() < self.max_users:
            user_id, prompt, params = prompt_q.get()

            # Cancel on special stop token
            if prompt == "<|stop|>":
                if any(
                    (user is not None) and (user_id == user.user_id)
                    for user in self.users
                ):
                    logger.info(f"Cancelling input from user {user_id}")
                    self._get_user_by_id(user_id).cancel = True
                else:
                    logger.info(f"Unexpected cancelling for non-activte user {user_id}")
                continue

            # Don't accept a prompt from a user that's already being procesed
            if any(
                (user is not None) and (user_id == user.user_id) for user in self.users
            ):
                logger.warning(f"Ignoring duplicate input from user {user_id}")
                continue

            user_info = UserInfo(user_id, prompt, 0, params, self.tokenizer)
            idx = self._find_free_user_slot()
            self.users[idx] = user_info
            if self.verbose:
                logger.debug(
                    f"Added user {user_id} to slot {idx} with prompt: {prompt}"
                )

    def pick_prompts(self, prompt_q: Queue):
        if self._get_num_of_users() == self.max_users:
            return

        if self._get_num_of_users() == 0:
            while prompt_q.empty():
                time.sleep(0.02)
            self._add_users_from_non_empty_queue(prompt_q)

        else:
            if prompt_q.empty():
                return
            else:
                self._add_users_from_non_empty_queue(prompt_q)

        # Check for duplicate user_ids and log it
        user_ids = [user.user_id for user in self.users if user is not None]
        if len(user_ids) != len(set(user_ids)):
            logger.warning(f"WARNING: Duplicate user ids: {user_ids}")

    def prepare_inputs(self):
        input_prompts = [user_info.prompt for user_info in self.users if user_info]
        self.prefill_ids, self.num_users, self.num_input_tokens = preprocess_and_validate_inputs(
            input_prompts, self.tokenizer, self.max_seq_len
        )

    def prefill(self):
        logger.info("Running prefill ...")
        self.timer_start("prefill")
        if self.prefill_on_host:
            logger.info("Initializing and filling KV cache")
            profiler.start(f"initializing_KV_cache_on_host")
            pt_logits, kv_cache = initialize_and_fill_kv_cache(
                self.pytorch_FalconCausalLM,
                self.model_config,
                self.configuration,
                self.prefill_ids[:, :self.num_input_tokens],
                self.num_layers,
                self.batch_size,
                self.max_seq_len,
                self.devices,
            )
            profiler.end(f"initializing_KV_cache")

            output_ids = torch.zeros(self.num_users, 1, dtype=torch.int64)
            for user_id in range(self.num_users):
                user_output_ids = self.post_processor(logits=pt_logits[user_id : user_id + 1, :, :], index=num_input_tokens - 1)
                output_ids[user_id] = user_output_ids

        if not self.prefill_on_host:
            logger.info("Running 1st run prefill stage with compile...")
            output_ids = torch.zeros(self.num_users, 1, dtype=torch.int64)
            
            for user_id in range(self.num_users):
                logger.info(f"Filling kv cache for user {user_id + 1}")
                (
                    tt_prefill_embeddings,
                    tt_prefill_attention_mask,
                ) = self.tt_FalconCausalLM.model_preprocessing(
                    "prefill", self.prefill_ids[user_id : user_id + 1], 0, num_input_tokens=num_input_tokens
                )
                assert tt_prefill_attention_mask is not None

                tt_logits, kv_cache = self.tt_FalconCausalLM(
                    input_embeddings=tt_prefill_embeddings,
                    llm_mode="prefill",
                    attention_mask=tt_prefill_attention_mask,
                    user_id=user_id,
                    layer_past=kv_cache,
                    layer_past_len=0,
                    use_cache=self.use_cache,
                )
                # time_prefill_compile_end = time.time()
                # time_prefill_compile += time_prefill_compile_end - time_prefill_compile_start

                # TODO: same as .deallocate()?
                del tt_prefill_embeddings
                del tt_prefill_attention_mask

                logits = torch.cat([tt2torch_tensor(tt_o).squeeze(1) for tt_o in tt_logits], -1)
                del tt_logits

                user_output_ids = self.post_processor(logits=logits, index=self.num_input_tokens - 1)
                output_ids[user_id] = user_output_ids
            self.timer_start("prefill")

        # TODO: Should the concat be removed since output token for prefill shouldn't be used
        self.generated_ids = torch.concat((self.prefill_ids[..., :self.num_input_tokens], output_ids), dim=1)

        for device in self.devices:
            tt_lib.device.Synchronize(device)
        logger.info("Done prefill")

    def decode(self):
        self.timer_stop("all_but_decode")
        self.timer_start("decode_preprocessing")
        (
            tt_decode_embeddings_host,
            tt_decode_attention_mask_host,
        ) = self.tt_FalconCausalLM.model_preprocessing("decode", self.decode_ids, self.kv_cache_len, num_input_tokens=self.kv_cache_len + 1)
        assert tt_decode_attention_mask_host is not None
        tt_decode_embeddings = [
            tt_decode_embeddings_host[i].to(self.devices[i], self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"])
            for i in range(len(self.devices))
        ]
        tt_decode_attention_mask = [
            tt_decode_attention_mask_host[i].to(self.devices[i], self.attention_mask_memconfig) for i in range(len(self.devices))
        ]
        # TODO: can preprocessing happen on device?
        # (
        #     tt_decode_embeddings,
        #     tt_decode_attention_mask,
        # ) = self.tt_FalconCausalLM.model_preprocessing("decode", decode_ids, self.kv_cache_len, num_input_tokens=self.kv_cache_len + 1)
        # assert tt_decode_attention_mask is not None
        self.timer_stop("decode_preprocessing")
        self.timer_start("decode")
        tt_logits, kv_cache = self.tt_FalconCausalLM(
            input_embeddings=tt_decode_embeddings,
            llm_mode="decode",
            attention_mask=tt_decode_attention_mask,
            layer_past=kv_cache,
            layer_past_len=self.kv_cache_len,
            use_cache=self.use_cache,
        )
        # TODO: does del work the same as .deallocate()?
        del tt_decode_embeddings
        if tt_decode_attention_mask is not None:
            del tt_decode_attention_mask

        # tt_outs = []
        logits = torch.cat([tt2torch_tensor(tt_o).squeeze(1) for tt_o in tt_logits], -1)
        del tt_logits
        self.timer_stop("decode")

        # decode_ids = self.post_processor(logits=logits, index=...).reshape(self.batch_size, 1)

        # for user_id, user_decode_id in enumerate(decode_ids[:self.num_users]):
        #     if user_decode_id == END_OF_TEXT:
        #         self.prompt_is_done[user_id] = True
        #     if self.prompt_is_done[user_id]:
        #         decode_ids[user_id] = SPACE

        self.timer_start("token_selection")
        self.timer_start("batch_top_pk_logits_efficient")
        self.decode_ids = batch_top_pk_logits_efficient(
            logits,
            top_ps=self.get_user_param("top_p"),
            top_ks=self.get_user_param("top_k"),
            temperatures=self.get_user_param("temperature"),
        ).reshape(self.batch_size, 1)
        self.timer_stop("batch_top_pk_logits_efficient")

        for idx, user_decode_id in enumerate(self.decode_ids):
            if self.users[idx] is None:
                continue
            self.users[idx].num_tokens_generated += 1
            if user_decode_id == self.tokenizer.eos_token_id:
                self.users[idx].decode_complete = True
            elif self.users[idx].num_tokens_generated > self.users[idx].max_tokens:
                self.users[idx].decode_complete = True
            elif (self.users[idx].stop_sequence is not None) and (
                user_decode_id == self.users[idx].stop_sequence
            ):
                self.users[idx].decode_complete = True

            if self.users[idx].decode_complete:
                self.decode_ids[idx] = self.tokenizer.eos_token_id

        self.generated_ids = torch.concat((self.generated_ids, self.decode_ids[:self.num_users]), dim=1)      
        self.timer_stop("token_selection")
        self.kv_cache_len += 1
        self.timer_start("all_but_decode")

    def push_outputs(self, output_q):
        for i, token_id in enumerate(self.decode_ids):  # bc input_ids is 1x32
            if self.users[i] is None:
                continue
            push_token_ids = []
            push_token_ids.append(token_id.item())
            return_text = self.tokenizer.decode(
                push_token_ids, clean_up_tokenization_spaces=True
            )
            output_q.put((self.users[i].user_id, return_text))
            if self.verbose:
                logger.debug(f"user_id:{self.users[i].user_id}, {return_text}")

    def reset_user_memory(self, user_idx, user):
        self.decode_ids[user_idx, 0] = 0

    def update_users(self):
        for i, token_id in enumerate(self.decode_ids):  # bc input_ids is 1x32
            if self.users[i] is None:
                continue

            if (
                token_id == self.tokenizer.eos_token_id
                and self.users[i].decode_complete
            ):
                self.reset_user_memory(i, self.users[i])
                self.users[i] = None
                if self.verbose:
                    logger.debug(
                        f"Evicted user_id: {self.users[i].user_id} from index {i} in user list"
                    )
            elif (
                token_id == self.tokenizer.eos_token_id
                and not self.users[i].decode_complete
            ):
                logger.error(
                    f"user_id: {self.users[i].user_id} from index {i} had EOS token but decode_complete=False."
                )
                self.reset_user_memory(i, self.users[i])
                self.users[i] = None
            elif (
                token_id != self.tokenizer.eos_token_id
                and self.users[i].decode_complete
            ):
                logger.error(
                    f"user_id: {self.users[i].user_id} from index {i} did not have EOS token but decode_complete=True."
                )
                self.reset_user_memory(i, self.users[i])
                self.users[i] = None

    def send_status(self, prompt_q, status_q):
        if time.time() - self.time_last_status > self.update_period:
            # send status queue which includes the (length of the prompt_q, the number of users being decoded rn, the user_ids being decoded)
            cur_status = (
                prompt_q.qsize(),
                self._get_num_of_users(),
                [user.user_id for user in self.users if user is not None],
            )
            status_q.put(cur_status)
            # udpate cur time
            self.time_last_status = time.time()

    def run_generate(self, prompt_q, output_q, status_q):
        """
        Continuously pop prompt from prompt_q and push generated tokens to output_q
        while running decode. Automatically swap users from prompt_q
        prompt_q: {'user_id1': 'prompt1', 'user_id2': 'prompt2'...}
        output_q: {'user_id1': 'generated_1', 'user_id3': 'generated_1', 'user_id1': 'generated_2'...}
        """
        logger.info("starting run_generate ...")
        while True:
            if self.verbose:
                logger.debug(f"run_generate step: {self.num_steps}")
            self.pick_prompts(prompt_q)  # we update to self.users
            self.prepare_inputs()
            if any([not user.prefill_complete for user in self.get_users()]):
                self.prefill()
            logger.info("Running inference decode and pushing results ...")
            while not all([user.decode_complete for user in self.get_users()]):
                self.decode()
                self.push_outputs(output_q)
                self.update_users()
                self.send_status(prompt_q, status_q)
            self.num_steps += 1


def batch_top_pk_logits_efficient(
    logits,
    top_ps=[0.9],
    top_ks=[10],
    temperatures=[1.0],
    return_probs=False,
    skip_token=11,
):
    out_tokens = []
    for b_logits, p, k, temperature in zip(logits[0], top_ps, top_ks, temperatures):
        if p is None:
            # skip None users
            token = torch.tensor([skip_token])
        else:
            # do not keep the entire vocab size after top k. Instead, keep the k size tensor and record the associated indices
            top_k_values, top_k_indices = torch.topk(b_logits.unsqueeze(0), k=k)
            # replace any nans with 0's
            top_k_values = torch.where(
                torch.isnan(top_k_values), torch.zeros_like(top_k_values), top_k_values
            )
            top_p_values = top_k_top_p_filtering(top_k_values, top_p=p)
            probs = F.softmax(top_p_values / temperature, dim=-1)
            top_k_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            token = top_k_indices.gather(-1, top_k_id.unsqueeze(-1)).squeeze(-1)

        out_tokens.append(token)
    return torch.concat(out_tokens)


def run_backend(prompt_q, output_q, status_q, verbose=True):
    logger.info("starting run_backend ...")
    with torch.no_grad():
        backend = PrefillDecodeBackend(
            model_version=inference_config.falcon_config.model_version,
            batch_size=inference_config.falcon_config.batch_size,
            num_layers=inference_config.falcon_config.num_layers,
            max_seq_len=inference_config.falcon_config.max_seq_len,
            cache_root=inference_config.cache_root,
            verbose=verbose,
        )
        try:
            # run generate
            backend.run_generate(prompt_q, output_q, status_q)
        except Exception as e:
            logger.error(e)
            # Capture the stack trace
            stack_trace = traceback.format_exc()
            logger.error(stack_trace)
            # Re-raise the exception if you want the process to exit with an error
            raise e
        finally:
            backend.teardown()
