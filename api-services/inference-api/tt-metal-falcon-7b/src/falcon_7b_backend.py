import json
import os
import random
from argparse import ArgumentParser
from multiprocessing import Queue
from time import sleep, time

import json
import pytest
from functools import partial
import torch
from loguru import logger
import time
from pathlib import Path
from transformers import AutoTokenizer
import os


import traceback

import torch
import torch.nn.functional as F

if not os.environ.get("MOCK_MODEL"):
    import tt_lib as ttl
    from tt_metal_impl.tt.falcon_causallm import TtFalconCausalLM
    from tt_metal_impl.reference.hf_modeling_falcon import FalconConfig, FalconForCausalLM
    from tt_metal_impl.tt.model_config import get_model_config, get_tt_cache_path, model_config_entries
    from tt_metal_impl.utility_functions import (
        disable_compilation_reports,
        disable_persistent_kernel_cache,
        enable_persistent_kernel_cache,
        profiler,
        torch2tt_tensor,
        tt2torch_tensor,
    )
    from transformers.generation.utils import top_k_top_p_filtering

from inference_config import inference_config
from inference_logger import get_logger

logger = get_logger(__name__)
logger.info(f"importing {__name__}")

END_OF_TEXT = 11
SPACE = 204

def preprocess_and_validate_inputs(input_prompts, tokenizer, max_seq_len):
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        input_prompts, padding="max_length", max_length=max_seq_len, add_special_tokens=False, return_tensors="pt"
    )
    prefill_ids = tokenized_inputs["input_ids"]

    tokenized_inputs_nopad = tokenizer(
        input_prompts, padding=False, max_length=max_seq_len, add_special_tokens=False, return_tensors="pt"
    )

    num_users = len(tokenized_inputs_nopad["input_ids"])
    num_input_tokens = len(tokenized_inputs_nopad["input_ids"][0])
    for input_prompt in tokenized_inputs_nopad["input_ids"]:
        assert len(input_prompt) == num_input_tokens
    logger.info(f"# of users: {num_users}")
    logger.info(f"# of input tokens per user: {num_input_tokens}")

    return prefill_ids, num_users, num_input_tokens

def initialize_kv_cache(configuration, num_layers, batch_size, max_seq_len, device):
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    kv_cache = ()
    for _ in range(num_layers):
        k_cache = torch.zeros(batch_size, 1, max_seq_len, head_dim)
        v_cache = torch.zeros(batch_size, 1, max_seq_len, head_dim)
        tt_k_cache = torch2tt_tensor(k_cache, device)
        tt_v_cache = torch2tt_tensor(v_cache, device)
        kv_cache += ((tt_k_cache, tt_v_cache),)
    return kv_cache

class PrefillDecodeBackend:
    # class UserInfo:
    #     def __init__(self, user_id, prompt, position_id, params, tokenizer):
    #         self.user_id = user_id
    #         self.prompt = prompt
    #         self.position_id = position_id
    #         tokenized = tokenizer(
    #             prompt, padding="max_length", return_tensors="pt", max_length=2048
    #         )
    #         # remove any EOS tokens for input
    #         tokenized.input_ids = tokenized.input_ids[
    #             tokenized.input_ids != tokenizer.eos_token_id
    #         ]
    #         # pad back to 2048 tokens
    #         tokenized.input_ids = F.pad(
    #             tokenized.input_ids,
    #             (0, 2048 - tokenized.input_ids.size(0)),
    #             "constant",
    #             0,
    #         )

    #         self.prompt_tokens = tokenized.input_ids.clone().squeeze()  # (2048,)
    #         self.prompt_length = torch.sum(tokenized.attention_mask).item()  # int

    #         self.generation_params = params
    #         self.max_tokens = params["max_tokens"]
    #         self.return_prompt = params["return_prompt"]
    #         self.cancel = False
    #         self.stop_sequence = None
    #         # Create backend log dir
    #         self.backend_log_dir = os.path.join(
    #             inference_config.log_cache, "backend_logs"
    #         )
    #         if not os.path.exists(self.backend_log_dir):
    #             os.mkdir(self.backend_log_dir)
    #         if params.get("stop_sequence"):
    #             self.stop_sequence = tokenizer(params.get("stop_sequence")).input_ids[0]

    def __init__(self, model_version, batch_size, num_layers, max_seq_len, cache_root, verbose=False) -> None:
        """
        Initialize pybuda model and all infracstructures to continuously run decode
        Maintain a cur_prompts for decode.
        """
        # self.max_users = args.user_rows
        # self.users = [None for _ in range(args.user_rows)]
        # self.seqlen = args.seqlen
        # self.fracture_vocab_factor = args.fracture_vocab_factor
        # self.fracture_vocab = args.fracture_vocab
        # self.device = args.device
        # self.num_layers = args.num_layers
        # self.top_p = args.top_p
        # self.top_k = args.top_k
        # self.temperature = args.temperature

        # # inputs to model
        # self.input_ids = torch.zeros((1, self.max_users), dtype=torch.long)
        # self.position_ids = None
        # self.attention_mask = None
        # self.kv_mask_id = None

        # # load model and tokenizer
        # self.model, self.tokenizer = self.load_model_and_tokenizer(args)
        # self.tokenizer.pad_token_id = 0

        # # kv_cache
        # self.past_key_values = self._init_kv_cache()  # TODO implement this
        # self._post_init_pybudify(args)

        # # decode backend status
        # self.cur_time = time()
        # self.update_period = 1

        # # Track number of loop iterations
        # self.num_steps = 0

        # self.verbose = verbose

        # if self.verbose:
        #     # Log all initialization
        #     with open(f"{self.self.backend_log_dir}/decode_backend.txt", "a") as f:
        #         f.write("\n\n" + "=" * 20 + "\n")
        #         f.write("Decode Backend Initialized\n")
        #         f.write("=" * 20 + "\n\n")
        #         f.write("Arguments:\n")
        #         f.write(json.dumps(vars(args), indent=4))
        #         f.write("\n\n")

        # new init:
        breakpoint()
        self.model_version = model_version
        # self.device = device
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.model_config = get_model_config("BFLOAT16-DRAM")
        #
        self.device = None
        self.cache_root = Path(cache_root)
        if not self.cache_root.exists():
            self.cache_root.mkdir(parents=True, exist_ok=True)
        # initialization            
        if not os.environ.get("MOCK_MODEL"):
            self.init_tt_metal()

    def model_location_generator(self, model_version, model_subdir=""):
        model_cache_path = Path(self.cache_root) / "tt-metal-models" / model_version
        model_cache_path.mkdir(parents=True, exist_ok=True)
        return model_cache_path

    def get_tt_cache_path(self, model_version, model_subdir="", default_dir=""):
        tt_cache_path = Path(self.cache_root) / "tt-metal-cache" / model_version
        tt_cache_path.mkdir(parents=True, exist_ok=True)
        return tt_cache_path

    def teardown(self):
        print("teardown ...")
        logger.info("teardown ...")
        if not os.environ.get("MOCK_MODEL"):
            self.teardown_tt_metal_device()

    def teardown_tt_metal_device(self):
        logger.info("teardown_tt_metal_device ...")
        ttl.device.CloseDevice(self.device)
        ttl.device.DeallocateBuffers(self.device)

    def init_tt_metal_device(self):
        logger.info("init_tt_metal_device ...")
        # TODO: can this be determined?
        # if not, use environment var
        device_id = os.getenv("DEVICE_ID", 0)
        device = ttl.device.CreateDevice(device_id)
        ttl.device.SetDefaultDevice(device)
        self.device = ttl.device.GetDefaultDevice()
        

    def init_tt_metal(self):
        logger.info("init_tt_metal ...")
        self.init_tt_metal_device()
        ttl.program_cache.enable()
        disable_persistent_kernel_cache()
        disable_compilation_reports()

        torch.manual_seed(0)

        tt_cache_path = self.get_tt_cache_path(self.model_version)

        configuration = FalconConfig(**model_config_entries)

        # State dict is needed for embeddings
        logger.info("Loading weights...")
        profiler.start(f"loading_weights")
        if len(os.listdir(tt_cache_path)) < 260:
            logger.info("Weights not found on machine; downloading weights...")
            model_cache = self.model_location_generator(self.model_version)
            # use cache_dir arg
            hugging_face_reference_model = FalconForCausalLM.from_pretrained(self.model_version, low_cpu_mem_usage=True, cache_dir=model_cache)
            hugging_face_reference_model.eval()
            state_dict = hugging_face_reference_model.state_dict()
            torch.save(state_dict["transformer.word_embeddings.weight"], tt_cache_path / "embedding.pt")
        else:
            state_dict = None

        logger.info("Loading weights finished!")
        profiler.end(f"loading_weights")

        ttl.device.Synchronize(self.device)

        logger.info("Moving weights to device; might take some time...")
        profiler.start(f"moving_to_device")

        base_url = ""
        self.tt_FalconCausalLM = TtFalconCausalLM(
            self.device,
            state_dict,
            base_url,
            self.num_layers,
            configuration,
            self.max_seq_len,
            self.model_config,
            tt_cache_path,
        )

        logger.info("Moved weights to device!")
        profiler.end(f"moving_to_device")

        ttl.device.Synchronize(self.device)

        logger.info("Tokenizing inputs...")
        profiler.start(f"tokenizing_inputs")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_version)

        profiler.end(f"tokenizing_inputs")

        logger.info("Initializing KV cache...")
        profiler.start(f"initializing_KV_cache")
        self.kv_cache = initialize_kv_cache(configuration, num_layers, batch_size, max_seq_len, device)
        profiler.end(f"initializing_KV_cache")
        profiler.disable()

    # def load_model_and_tokenizer(self, args):
    #     pass

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

            user_info = DecodeBackend.UserInfo(
                user_id, prompt, 0, params, self.tokenizer
            )
            idx = self._find_free_user_slot()
            self.users[idx] = user_info
            if self.verbose:
                with open(f"{self.self.backend_log_dir}/{user_id}.txt", "a") as f:
                    f.write("\n<Prompt>: " + prompt + "\n")
                with open(f"{self.self.backend_log_dir}/decode_backend.txt", "a") as f:
                    f.write(
                        f"Added user {user_id} to slot {idx} with prompt: {prompt}\n"
                    )

    def pick_prompts(self, prompt_q: Queue):
        if self._get_num_of_users() == self.max_users:
            return

        if self._get_num_of_users() == 0:
            while prompt_q.empty():
                sleep(0.02)
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
        profiler.start(f"loading_inputs")
        if len(user_input) == 1:
            input_prompts = user_input
        else:
            input_prompts = load_inputs(user_input, batch_size)

        profiler.end(f"loading_inputs")
        
        prefill_ids, num_users, num_input_tokens = preprocess_and_validate_inputs(input_prompts, self.tokenizer, self.max_seq_len)



    # def prepare_inputs(self):
    #     """
    #     input_ids:          1x32, predictions (long)
    #     prompt_tokens:      32xseqlen padded prompts (long)
    #     position_ids:       1x32 (long) tensor of position_ids for the currently generated token
    #     prompt_lengths:     32, tensor of number of tokens in each prompt
    #     attention_mask:     1x32x2048
    #     kv_mask_id:         1x32
    #     """

    #     # create tensor position_ids based on user_info.position_id
    #     position_ids = [
    #         user_info.position_id if user_info is not None else 0
    #         for user_info in self.users
    #     ]
    #     self.position_ids = torch.tensor(position_ids, dtype=torch.long).unsqueeze(0)

    #     # create attention mask and kv_mask_id based on position_id
    #     attention_mask = torch.zeros(
    #         (1, self.max_users, self.seqlen), dtype=torch.int32
    #     )

    #     for i, pos_id in enumerate(self.position_ids[0]):
    #         attention_mask[0, i, :pos_id] = 1

    #     kv_mask_id = (
    #         self.position_ids.clone().view(1, -1).to(torch.int32)
    #     )  # batch, num_users
    #     self.attention_mask = attention_mask
    #     self.kv_mask_id = kv_mask_id

    #     # if the position id is 0, then the input_id is always the first token
    #     for i in range(self.max_users):
    #         if self.users[i] is not None and self.position_ids[0, i] == 0:
    #             self.input_ids[0, i] = self.users[i].prompt_tokens[0]

    #     # make a tensor of prompt_tokens
    #     prompt_tokens = [
    #         (
    #             user_info.prompt_tokens
    #             if user_info is not None
    #             else torch.zeros((self.seqlen), dtype=torch.long)
    #         )
    #         for user_info in self.users
    #     ]
    #     prompt_lengths = [
    #         user_info.prompt_length if user_info is not None else 0
    #         for user_info in self.users
    #     ]
    #     self.prompt_tokens = torch.stack(prompt_tokens)

    #     self.prompt_lengths = torch.tensor(prompt_lengths)

    #     if self.verbose:
    #         torch.set_printoptions(threshold=10000)
    #         with open(f"{self.self.backend_log_dir}/decode_backend.txt", "a") as f:
    #             f.write(f"\nprepare_inputs()\n")
    #             f.write(f"input_ids: {self.input_ids}\n")
    #             f.write(f"position_ids: {self.position_ids}\n")
    #             f.write(f"prompt_tokens: {self.prompt_tokens}\n")
    #             f.write(f"prompt_lengths: {self.prompt_lengths}\n")
    #             f.write(f"attention_mask: {self.attention_mask}\n")
    #             f.write(f"kv_mask_id: {self.kv_mask_id}\n")
    #             f.write(
    #                 f"user valid bitmask: {[1 if user is not None else 0 for user in self.users]}"
    #             )

    def sequential_prefill_decode(self):
        logger.info("Running prefill ...")
        post_processor = partial(post_process)
        use_cache = True
        output_ids = torch.zeros(num_users, 1, dtype=torch.int64)
        time_prefill_compile = 0
        for user_id in range(num_users):
            time_prefill_compile_start = time.time()
            (
                tt_prefill_embeddings,
                tt_prefill_attention_mask,
            ) = self.tt_FalconCausalLM.model_preprocessing(
                "prefill", prefill_ids[user_id : user_id + 1], 0, num_input_tokens=num_input_tokens
            )
            assert tt_prefill_attention_mask is not None

            tt_logits, kv_cache = self.tt_FalconCausalLM(
                input_embeddings=tt_prefill_embeddings,
                llm_mode="prefill",
                attention_mask=tt_prefill_attention_mask,
                user_id=user_id,
                layer_past=kv_cache,
                layer_past_len=0,
                use_cache=use_cache,
            )
            time_prefill_compile_end = time.time()
            time_prefill_compile += time_prefill_compile_end - time_prefill_compile_start

            tt_prefill_embeddings.deallocate()
            if tt_prefill_attention_mask is not None:
                tt_prefill_attention_mask.deallocate()

            logits = tt2torch_tensor(tt_logits).squeeze(1)
            tt_logits.deallocate()

            user_output_ids = post_processor(logits=logits, index=num_input_tokens - 1)
            output_ids[user_id] = user_output_ids

        generated_ids = torch.concat((prefill_ids[..., :num_input_tokens], output_ids), dim=1)

        ttl.device.Synchronize(device)
        logger.info("Done prefill")

        logger.info("Running inference decode stage...")
        decode_ids = torch.zeros(batch_size, 1, dtype=torch.int64)
        for user_id, output_id in enumerate(output_ids):
            decode_ids[user_id] = output_id

        kv_cache_len = num_input_tokens  # This will increment by one after each decode
        prompt_is_done = [False for _ in range(num_users)]

        time_decode_inference = 0
        for output_token_index in range(max_seq_len - num_input_tokens):
            time_decode_inference_start = time.time()
            (
                tt_decode_embeddings,
                tt_decode_attention_mask,
            ) = tt_FalconCausalLM.model_preprocessing("decode", decode_ids, kv_cache_len, num_input_tokens=kv_cache_len + 1)
            assert tt_decode_attention_mask is not None

            tt_logits, kv_cache = tt_FalconCausalLM(
                input_embeddings=tt_decode_embeddings,
                llm_mode="decode",
                attention_mask=tt_decode_attention_mask,
                layer_past=kv_cache,
                layer_past_len=kv_cache_len,
                use_cache=use_cache,
            )
            time_decode_inference_end = time.time()
            time_decode_inference += time_decode_inference_end - time_decode_inference_start

            tt_decode_embeddings.deallocate()
            if tt_decode_attention_mask is not None:
                tt_decode_attention_mask.deallocate()

            logits = tt2torch_tensor(tt_logits).squeeze(1)
            tt_logits.deallocate()

            decode_ids = post_processor(logits=logits, index=...).reshape(batch_size, 1)

            for user_id, user_decode_id in enumerate(decode_ids[:num_users]):
                if user_decode_id == END_OF_TEXT:
                    prompt_is_done[user_id] = True
                if prompt_is_done[user_id]:
                    decode_ids[user_id] = SPACE

            if all(prompt_is_done):
                break

            generated_ids = torch.concat((generated_ids, decode_ids[:num_users]), dim=1)
            kv_cache_len += 1

        logger.info("Finished inference decode stage!")
        


    def decode(self):
        outputs = self.model.main_forward_part(
            self.input_ids,
            position_ids=self.position_ids,
            attention_mask=self.attention_mask,
            kv_mask_id=self.kv_mask_id,
            past_key_values=self.past_key_values,
            profile_runtime=False,
            on_device_lm_head=True,
        )

        output = (
            outputs[0]
            if not self.fracture_vocab
            else torch.cat(outputs[: self.fracture_vocab_factor], dim=-1)
        )
        output = output.to("cpu").squeeze()
        past_key_values_cpy = (
            outputs[1:]
            if not self.fracture_vocab
            else outputs[self.fracture_vocab_factor :]
        )

        # take only up to vocab_size logits in case padding
        output = output[..., : self.model.vocab_size]

        if self.device == "pytorch":
            # update kv cache
            # past_key_values is [key1_l1, ... key8_l1, value1_l1, ... value8_l1, key1_l2, ...]
            # shape is already good: [1, 32, 2048, 64]
            # change it to [
            new_past_key_values = []
            for i in range(self.num_layers):
                # 8ks and 8vs for each layer -> 16 for each layer
                new_past_key_values.append(past_key_values_cpy[i * 16 : (i + 1) * 16])
            self.past_key_values = tuple(new_past_key_values)

        if not self.top_p:  # greedy
            output_tokens = output.argmax(dim=-1)
        else:
            top_ps = [
                user.generation_params["top_p"] if user is not None else self.top_p
                for user in self.users
            ]
            top_ks = [
                user.generation_params["top_k"] if user is not None else self.top_k
                for user in self.users
            ]
            temperatures = [
                (
                    user.generation_params["temperature"]
                    if user is not None
                    else self.temperature
                )
                for user in self.users
            ]
            output_tokens = batch_top_pk_logits_efficient(
                output.to(torch.float32), top_ps, top_ks, temperatures
            )

        # if user has hit max_length, send eos token
        for idx, user in enumerate(self.users):
            if user is not None:
                if (user.position_id - user.prompt_length + 1) >= user.max_tokens:
                    output_tokens[idx] = self.tokenizer.eos_token_id
                elif (
                    (user.stop_sequence is not None)
                    and (user.position_id - user.prompt_length + 1) > 0
                    and (output_tokens[idx] == user.stop_sequence)
                ):
                    output_tokens[idx] = self.tokenizer.eos_token_id

        # update the new tokens generated to the input id
        self.input_ids = output_tokens.view(1, self.max_users)

        if self.verbose:
            with open(f"{self.self.backend_log_dir}/decode_backend.txt", "a") as f:
                f.write(f"\ndecode()\n")
                f.write(f"outputs: {self.input_ids}\n")
                f.write(
                    f"decoded outputs: {[self.tokenizer.decode(input_id, clean_up_tokenization_spaces=True) for input_id in self.input_ids[0]]}\n"
                )

    def switch_tokens(self):
        """
        This function either picks up the next prompt token or the current prediction,
        depending on whether we have completed prefill or not.

        self.input_ids:          1x32, predictions (long)
        self.prompt_tokens:      32xseqlen padded prompts (long)
        self.position_ids:       1x32 (long) tensor of position_ids for the currently generated token
        self.prompt_lengths:     32, tensor of number of tokens in each prompt
        """
        output_ids = torch.zeros(1, 32, dtype=torch.long)
        for i in range(self.input_ids.size(1)):
            if self.users[i] is None:
                continue
            next_input_idx = self.position_ids[0, i] + 1
            if next_input_idx >= self.prompt_lengths[i]:
                output_ids[0, i] = self.input_ids[0, i]
            else:
                output_ids[0, i] = self.prompt_tokens[i][next_input_idx]
            if self.users[i].cancel or next_input_idx >= self.seqlen - 1:
                output_ids[0, i] = self.tokenizer.eos_token_id

        self.input_ids = output_ids

        if self.verbose:
            with open(f"{self.self.backend_log_dir}/decode_backend.txt", "a") as f:
                f.write(f"\nswitch_tokens()\n")
                f.write(f"outputs: {self.input_ids}\n")
                f.write(
                    f"decoded outputs: {[self.tokenizer.decode(input_id, clean_up_tokenization_spaces=True) for input_id in self.input_ids[0]]}\n"
                )

    def push_outputs(self, output_q):
        for i, token in enumerate(self.input_ids[0]):  # bc input_ids is 1x32
            if self.users[i] is None:
                continue
            push_tokens = []
            # optionally respond with prompt as it is prefilled
            if not self.users[i].return_prompt and self.users[i].position_id < (
                self.users[i].prompt_length - 1
            ):
                continue
            elif self.users[i].position_id == 0:
                # return 0th prompt token with 1st prompt token in the first push
                push_tokens.append(self.users[i].prompt_tokens[0])

            push_tokens.append(token)
            return_text = self.tokenizer.decode(
                push_tokens, clean_up_tokenization_spaces=True
            )
            output_q.put((self.users[i].user_id, return_text))

            if self.verbose:
                # Log user's output
                with open(
                    f"{self.self.backend_log_dir}/{self.users[i].user_id}.txt", "a"
                ) as f:
                    f.write(return_text)
                with open(f"{self.self.backend_log_dir}/decode_backend.txt", "a") as f:
                    f.write(f"\npush_outputs()\n")
                    f.write(
                        f"pushing user_id: {self.users[i].user_id}, data: {return_text}\n"
                    )

    def update_users(self):
        for i, token in enumerate(self.input_ids[0]):  # bc input_ids is 1x32
            if self.users[i] is None:
                continue
            token_text = self.tokenizer.decode(token, clean_up_tokenization_spaces=True)
            if token_text == self.tokenizer.eos_token:
                if self.verbose:
                    with open(
                        f"{self.self.backend_log_dir}/decode_backend.txt", "a"
                    ) as f:
                        f.write(
                            f"\nEvicted user_id: {self.users[i].user_id} from index {i} in user list\n"
                        )

                self.users[i] = None
                # make input id 0 after ejection
                self.input_ids[0, i] = 0
            else:
                self.users[i].position_id += 1

    def send_status(self, prompt_q, status_q):
        if time() - self.cur_time > self.update_period:
            # send status queue which includes the (length of the prompt_q, the number of users being decoded rn, the user_ids being decoded)
            cur_status = (
                prompt_q.qsize(),
                self._get_num_of_users(),
                [user.user_id for user in self.users if user is not None],
            )
            status_q.put(cur_status)
            # udpate cur time
            self.cur_time = time()

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
                with open(f"{self.self.backend_log_dir}/decode_backend.txt", "a") as f:
                    f.write(f"\nLOOP ITERATION {self.num_steps}\n")
            self.pick_prompts(prompt_q)  # we update to self.users
            self.prepare_inputs()
            self.sequential_prefill_decode()
            # self.switch_tokens()
            self.push_outputs(output_q)
            self.update_users()
            self.send_status(prompt_q, status_q)
            self.num_steps += 1


def batch_top_pk_logits_efficient(
    logits, top_ps=[0.9], top_ks=[10], temperatures=[1.0], return_probs=False
):
    out_tokens = []
    for b_logits, p, k, temperature in zip(logits, top_ps, top_ks, temperatures):
        # do not keep the entire vocab size after top k. Instead, keep the k size tensor and record the associated indices
        top_k_values, top_k_indices = torch.topk(b_logits.unsqueeze(0), k=k)
        top_p_values = top_k_top_p_filtering(top_k_values, top_p=p)
        probs = F.softmax(top_p_values / temperature, dim=-1)
        top_k_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        token = top_k_indices.gather(-1, top_k_id.unsqueeze(-1)).squeeze(-1)
        if return_probs:
            # TODO: effectively batch probs
            raise NotImplementedError
            # return token, (probs, top_k_indices)
        else:
            out_tokens.append(token)
    return torch.concat(out_tokens)


def run_backend(prompt_q, output_q, status_q, arg_overrides, verbose=True):
    logger.error("starting run_backend ...")
    with torch.no_grad():
        
        # TODO: wire out these kwargs to arg_overrides, rename
        kwargs = {
            "model_version": "tiiuae/falcon-7b-instruct",
            "batch_size": 32,
            "num_layers": 32,
            "max_seq_len": 1024, 
            "cache_root": Path("/mnt/mldata/test_cache_root")
        }
        backend = PrefillDecodeBackend(**kwargs, verbose=verbose)
        try:
            # run generate
            backend.run_generate(prompt_q, output_q, status_q)
        except Exception as e:
            backend.teardown()
            # Capture the stack trace
            stack_trace = traceback.format_exc()
            # write the stack trace to the specified output file
            with open(f"{backend.backend_log_dir}/stack_trace", "w") as f:
                f.write(stack_trace)
            # Re-raise the exception if you want the process to exit with an error
            raise e
        finally:
            backend.teardown()
