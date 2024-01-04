import json
import os
import random
from argparse import ArgumentParser
from multiprocessing import Queue
from time import sleep, time

import torch
import torch.nn.functional as F
from decode_v0 import load_model_and_tokenizer
from pybudify40 import PyBudify
from transformers.generation.utils import top_k_top_p_filtering


class DecodeBackend:
    class UserInfo:
        def __init__(self, user_id, prompt, position_id, params, tokenizer):
            self.user_id = user_id
            self.prompt = prompt
            self.position_id = position_id
            tokenized = tokenizer(
                prompt, padding="max_length", return_tensors="pt", max_length=2048
            )
            self.prompt_tokens = tokenized.input_ids.clone().squeeze()  # (2048,)
            self.prompt_length = torch.sum(tokenized.attention_mask).item()  # int
            self.generation_params = params
            self.max_tokens = params["max_tokens"]
            self.return_prompt = params["return_prompt"]
            self.cancel = False
            self.stop_sequence = None
            if params.get("stop_sequence"):
                self.stop_sequence = tokenizer(params.get("stop_sequence")).input_ids[0]

    def __init__(self, args, verbose=False) -> None:
        """
        Initialize pybuda model and all infracstructures to continuously run decode
        Maintain a cur_prompts for decode.
        """
        self.max_users = args.user_rows
        self.users = [None for _ in range(args.user_rows)]
        self.seqlen = args.seqlen
        self.fracture_vocab_factor = args.fracture_vocab_factor
        self.fracture_vocab = args.fracture_vocab
        self.device = args.device
        self.num_layers = args.num_layers
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.temperature = args.temperature

        # inputs to model
        self.input_ids = torch.zeros((1, self.max_users), dtype=torch.long)
        self.position_ids = None
        self.attention_mask = None
        self.kv_mask_id = None

        # load model and tokenizer
        self.model, self.tokenizer = self.load_model_and_tokenizer(args)

        self.tokenizer.pad_token_id = 0

        # kv_cache
        self.past_key_values = self._init_kv_cache()  # TODO implement this

        self._post_init_pybudify(args)

        # decode backend status
        self.cur_time = time()
        self.update_period = 1

        # Track number of loop iterations
        self.num_steps = 0

        self.verbose = verbose

        if self.verbose:
            # Log all initialization
            with open("backend_logs/decode_backend.txt", "a") as f:
                f.write("\n\n" + "=" * 20 + "\n")
                f.write("Decode Backend Initialized\n")
                f.write("=" * 20 + "\n\n")
                f.write("Arguments:\n")
                f.write(json.dumps(vars(args), indent=4))
                f.write("\n\n")

    def load_model_and_tokenizer(self, args):
        return load_model_and_tokenizer(args)

    def _init_kv_cache(self):
        ext_past_kv = []
        # past_key_values is a tuple of [key, value] tensors, one for each layer
        # copy into the right parts of the key, value tensors

        for _ in range(self.num_layers):
            past_key = torch.zeros(
                (self.max_users, 8, self.seqlen, 64), dtype=torch.bfloat16
            )
            past_value = torch.zeros(
                (self.max_users, 8, self.seqlen, 64), dtype=torch.bfloat16
            )

            if self.device == "pytorch":
                past_key = past_key.to(torch.float32)
                past_value = past_value.to(torch.float32)

            ext_past_kv.append([past_key, past_value])

        past_key_values = tuple(ext_past_kv)

        if self.max_users > 1:
            new_past_key_values = []
            for k, v in past_key_values:
                # split the kv in each group into separate tensor
                # k, v are size [32 x 8 x 2048 x 64]
                layer_ks = []
                layer_vs = []
                for n in range(k.shape[1]):
                    # want it to be [1 x 32 x 2048 x 64]
                    layer_k = k[:, [n], :, :].transpose(0, 1)
                    layer_ks.extend([layer_k])
                    layer_v = v[:, [n], :, :].transpose(0, 1)
                    layer_vs.extend([layer_v])

                new_past_key_values.append(layer_ks + layer_vs)

            past_key_values = tuple(new_past_key_values)

        return past_key_values

    def _post_init_pybudify(self, args):
        # Now transition to running in token-by-token mode for generation
        if (
            args.device in ["pytorch", "golden", "silicon"]
            and args.version != "torch1.0"
        ):
            netlist_name = f'falcon_{"odlmh" if args.od_lm_head else ""}{"p" if args.padded_lm_head else ""}{"fv" if args.fracture_vocab else ""}_{"flash" if args.flash_decode else ""}_{args.num_chips}c_{args.fracture_mlp}mf_{args.fracture_attn}af_{args.num_layers}l_{args.seqlen}s_{args.net_name}'
            self.model.transformer.blocks = PyBudify(
                self.model.transformer.blocks,
                device=args.device,
                arch=args.arch,
                precision=args.precision,
                amp_level=args.amp_level,
                num_chips=args.num_chips,
                fuse=args.fuse,
                perf=args.perf,
                verify=args.verify,
                log_level=args.log_level,
                tti_load=args.load,
                tti_save=args.save,
                concurrent=(args.mode == "concurrent"),
                netlist_name=netlist_name,
                version=args.version,
                is_decode=True,
                multi_chip_placer=args.multi_chip_placer,
                fracture_attn=args.fracture_attn,
                fracture_mlp=args.fracture_mlp,
                enable_tvm_cache=args.enable_tvm_cache,
                opt_level=args.opt_level,
                num_tokens=args.num_tokens,
                user_rows=args.user_rows,
                queues_on_host=args.queues_on_host,
                od_lm_head=args.od_lm_head,
                fracture_vocab=args.fracture_vocab,
                fracture_vocab_factor=args.fracture_vocab_factor,
                flash_decode=args.flash_decode,
                num_outer_loops=args.num_outer_loops,
            )

        if args.version == "efficient-40b":
            if args.device != "pytorch":
                # assert args.mode == 'sequential', 'Only sequential mode supported for efficient-40b'
                self.model.transformer.blocks.to(torch.bfloat16)
        else:
            raise ValueError(f"Unknown version {args.version}")

        # Set up environment variables
        os.environ["PYBUDA_MICROBATCH_LOOPING"] = "1"
        os.environ["TT_BACKEND_DRAM_POLLING_FREQUENCY"] = "64"
        os.environ["PYBUDA_DEVICE_EMBEDDINGS"] = "1"
        os.environ["TT_BACKEND_ALLOW_RUNTIME_RECOMPILE"] = "1"
        os.environ["TT_BACKEND_COMPILE_THREADS"] = "32"

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
                    print(f"Cancelling input from user {user_id}")
                    self._get_user_by_id(user_id).cancel = True
                else:
                    print(f"Unexpected cancelling for non-activte user {user_id}")
                continue

            # Don't accept a prompt from a user that's already being procesed
            if any(
                (user is not None) and (user_id == user.user_id) for user in self.users
            ):
                print(f"Ignoring duplicate input from user {user_id}")
                continue

            user_info = DecodeBackend.UserInfo(
                user_id, prompt, 0, params, self.tokenizer
            )
            idx = self._find_free_user_slot()
            self.users[idx] = user_info
            if self.verbose:
                with open(f"backend_logs/{user_id}.txt", "a") as f:
                    f.write("\n<Prompt>: " + prompt + "\n")
                with open("backend_logs/decode_backend.txt", "a") as f:
                    f.write(
                        f"Added user {user_id} to slot {idx} with prompt: {prompt}\n"
                    )

    def pick_prompts(self, prompt_q: Queue):
        if self._get_num_of_users() == self.max_users:
            return

        if self._get_num_of_users() == 0:
            while prompt_q.empty():
                sleep(0.1)
            self._add_users_from_non_empty_queue(prompt_q)

        else:
            if prompt_q.empty():
                return
            else:
                self._add_users_from_non_empty_queue(prompt_q)

        # Check for duplicate user_ids and log it
        user_ids = [user.user_id for user in self.users if user is not None]
        if len(user_ids) != len(set(user_ids)):
            print(f"WARNING: Duplicate user ids: {user_ids}")

    def prepare_inputs(self):
        """
        input_ids:          1x32, predictions (long)
        prompt_tokens:      32xseqlen padded prompts (long)
        position_ids:       1x32 (long) tensor of position_ids for the currently generated token
        prompt_lengths:     32, tensor of number of tokens in each prompt
        attention_mask:     1x32x2048
        kv_mask_id:         1x32
        """

        # create tensor position_ids based on user_info.position_id
        position_ids = [
            user_info.position_id if user_info is not None else 0
            for user_info in self.users
        ]
        self.position_ids = torch.tensor(position_ids, dtype=torch.long).unsqueeze(0)

        # create attention mask and kv_mask_id based on position_id
        attention_mask = torch.zeros(
            (1, self.max_users, self.seqlen), dtype=torch.int32
        )

        for i, pos_id in enumerate(self.position_ids[0]):
            attention_mask[0, i, :pos_id] = 1

        kv_mask_id = (
            self.position_ids.clone().view(1, -1).to(torch.int32)
        )  # batch, num_users
        self.attention_mask = attention_mask
        self.kv_mask_id = kv_mask_id

        # if the position id is 0, then the input_id is always the first token
        for i in range(self.max_users):
            if self.users[i] is not None and self.position_ids[0, i] == 0:
                self.input_ids[0, i] = self.users[i].prompt_tokens[0]

        # make a tensor of prompt_tokens
        prompt_tokens = [
            user_info.prompt_tokens
            if user_info is not None
            else torch.zeros((self.seqlen), dtype=torch.long)
            for user_info in self.users
        ]
        prompt_lengths = [
            user_info.prompt_length if user_info is not None else 0
            for user_info in self.users
        ]
        self.prompt_tokens = torch.stack(prompt_tokens)

        self.prompt_lengths = torch.tensor(prompt_lengths)

        if self.verbose:
            torch.set_printoptions(threshold=10000)
            with open("backend_logs/decode_backend.txt", "a") as f:
                f.write(f"\nprepare_inputs()\n")
                f.write(f"input_ids: {self.input_ids}\n")
                f.write(f"position_ids: {self.position_ids}\n")
                f.write(f"prompt_tokens: {self.prompt_tokens}\n")
                f.write(f"prompt_lengths: {self.prompt_lengths}\n")
                f.write(f"attention_mask: {self.attention_mask}\n")
                f.write(f"kv_mask_id: {self.kv_mask_id}\n")
                f.write(
                    f"user valid bitmask: {[1 if user is not None else 0 for user in self.users]}"
                )

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
                user.generation_params["temperature"]
                if user is not None
                else self.temperature
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
            with open("backend_logs/decode_backend.txt", "a") as f:
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
            with open("backend_logs/decode_backend.txt", "a") as f:
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
                with open(f"backend_logs/{self.users[i].user_id}.txt", "a") as f:
                    f.write(return_text)
                with open("backend_logs/decode_backend.txt", "a") as f:
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
                    with open(f"backend_logs/decode_backend.txt", "a") as f:
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
        while True:
            if self.verbose:
                with open("backend_logs/decode_backend.txt", "a") as f:
                    f.write(f"\nLOOP ITERATION {self.num_steps}\n")
            self.pick_prompts(prompt_q)  # we update to self.users
            self.prepare_inputs()
            self.decode()
            self.switch_tokens()
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


def run_decode_backend(prompt_q, output_q, status_q, arg_overrides, verbose=False):
    args = parse_args(arg_overrides)

    # Create backend log dir
    if not os.path.exists("backend_logs"):
        os.mkdir("backend_logs")

    with torch.no_grad():
        # initialization
        decode_be = DecodeBackend(args, verbose=verbose)
        # run generate
        decode_be.run_generate(prompt_q, output_q, status_q)


def parse_args(inp=None):
    parser = ArgumentParser(
        "Generate text token-by-token starting with a pre-filled KV cache"
    )
    parser.add_argument(
        "-m", "--model", type=str, default="tiiuae/falcon-40b", help="Model name"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (number of model inputs, not necessarily number of users, see --user-rows)",
    )
    parser.add_argument(
        "--user-rows",
        type=int,
        default=1,
        help="Number of users packed into the rows of each input (affects pybuda devices only)",
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "concurrent", "huggingface"],
        help="Concurrent continuously streams batch-size users through the model, huggingface runs the unmodified model",
    )
    parser.add_argument(
        "-s", "--stop", type=str, default="\n\n", help="Text to stop decoding after"
    )
    parser.add_argument(
        "-n",
        "--num-tokens",
        type=int,
        default=10,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "-l",
        "--num-layers",
        type=int,
        default=60,
        help="Number of layers (default=all)",
    )
    parser.add_argument(
        "--output-at-end",
        action="store_true",
        help="Output at the end of generation instead of token by token",
    )
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length")

    parser.add_argument(
        "-d",
        "--device",
        choices=["cuda", "huggingface", "pytorch", "golden", "silicon"],
        default="huggingface",
        help="huggingface: run using HF code only, pytorch: use our shim but run in PyTorch, golden/silicon: run via pybuda",
    )
    parser.add_argument(
        "--arch",
        choices=["greyskull", "wormhole", "wormhole_b0", "galaxy", "nebula-galaxy"],
        default="wormhole_b0",
        help="Architecture to use for silicon",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16", "fp8", "fp8b"],
        default="fp32",
        help="Precision to use for all silicon tensors",
    )
    parser.add_argument(
        "--amp-level",
        type=int,
        choices=[0, 1, 2],
        help="Automatic mixed precision level (0=off, 1=mixed b-formats, 2=mixed a-formats)",
    )
    parser.add_argument(
        "--num-chips", type=int, default=1, help="Number of chips to use"
    )
    parser.add_argument(
        "--skip-lm-head",
        action="store_true",
        help="Skip the LM head and output garbage",
    )
    parser.add_argument(
        "--place",
        choices=["none", "single", "dual"],
        default="none",
        help="Manual placement of ops and buffers, default is none",
    )
    parser.add_argument("--fuse", action="store_true", help="Fuse layers")
    parser.add_argument(
        "--host-queues",
        action="store_true",
        help="Place input queues in host DRAM and let device read them over PCIe instead of copying them to device DRAM",
    )
    parser.add_argument(
        "--perf",
        choices=["none", "light", "verbose"],
        default=None,
        help="Performance tracing",
    )
    parser.add_argument("--verify", action="store_true", help="Verify results")
    parser.add_argument(
        "--log-level",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="ERROR",
        help="Log level",
    )
    parser.add_argument("--load", type=str, help="Load a TTImage")
    parser.add_argument("--save", type=str, help="Save a TTImage")

    parser.add_argument(
        "--version",
        type=str,
        choices=[
            "odkv_t",
            "odkv",
            "tt",
            "torch1.0",
            "split-mq",
            "efficient-40b",
            "debug",
        ],
        help="Version of the model to use",
    )
    parser.add_argument("--flash-decode", action="store_true", help="Flash decode mode")

    parser.add_argument(
        "-af",
        "--fracture-attn",
        type=int,
        default=0,
        help="Attention fracturing factor for version padded-fractured-full",
    )
    parser.add_argument(
        "-mf",
        "--fracture-mlp",
        type=int,
        default=0,
        help="MLP fracturing factor for version padded-fractured-full",
    )
    parser.add_argument(
        "--net-name", type=str, default="", help="Changes the netlist name"
    )

    parser.add_argument(
        "--load-pretrained", action="store_true", help="Load weights from hf."
    )
    # save and load weights because hf weights take way toooooo long to load
    parser.add_argument(
        "--save-weights",
        action="store_true",
        help="Save weights to disk. Used for debugging.",
    )
    parser.add_argument(
        "--save-weights-dir",
        type=str,
        default="falcon40b.pt",
        help="Directory to save weights to. Used for debugging.",
    )
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Load weights from disk. Used for debugging.",
    )
    parser.add_argument(
        "--load-weights-dir",
        type=str,
        default="falcon40b.pt",
        help="Directory to load weights from. Used for debugging.",
    )

    parser.add_argument(
        "--multi-chip-placer",
        type=str,
        choices=["parallel", "pipeline"],
        default="parallel",
        help="Parallel or pipeline placer for multi-chip systems",
    )
    parser.add_argument(
        "--opt-level", type=int, default=0, help="Runtime optimization level"
    )

    parser.add_argument(
        "--enable-tvm-cache", action="store_true", help="Enable TVM caching"
    )
    parser.add_argument(
        "--hf-cache",
        type=str,
        default="/proj_sw/user_dev/hf_data",
        help="Cache directory for huggingface",
    )

    parser.add_argument(
        "--queues-on-host",
        action="store_true",
        help='Set flag os.environ["PYBUDA_ENABLE_OUTPUT_QUEUES_ON_HOST"] = "0"',
    )
    parser.add_argument(
        "-odlmh", "--od-lm-head", action="store_true", help="Put LM head on device"
    )
    parser.add_argument(
        "-plmh",
        "--padded-lm-head",
        action="store_true",
        help="Put padded LM head on device",
    )
    parser.add_argument(
        "-fv", "--fracture-vocab", action="store_true", help="Fracture vocab"
    )
    parser.add_argument(
        "--fracture-vocab-factor", type=int, default=8, help="Fracture vocab factor"
    )
    parser.add_argument(
        "--top-p",
        default=None,
        type=float,
        help="Top p sampling threshold (default: greedy)",
    )
    parser.add_argument(
        "--top-k", default=10, type=int, help="Top k sampling threshold (default: 10)"
    )
    parser.add_argument(
        "-t",
        "--temperature",
        default=1.0,
        type=float,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--print-option",
        choices=["reprint", "output-to-file"],
        default="reprint",
        help="Print all options",
    )
    parser.add_argument("--print-interval", type=int, default=1, help="Print interval")
    parser.add_argument(
        "--num-outer-loops",
        type=int,
        default=1,
        help="Experimental: insert outer loop which calls run_generate this many times",
    )

    parser.add_argument("--prompts-file", type=str, default="data/multi_prompt.json")

    return parser.parse_args(inp)
