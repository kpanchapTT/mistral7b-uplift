import os
import threading
import warnings
from argparse import ArgumentParser
from queue import Queue
from time import time
from time import sleep
from multiprocessing import Queue

import torch
from pybudify40 import PyBudify
from multilineoutput import MultiLineOutput
from reprint import output as reprint_output
import torch.nn.functional as F
from transformers.generation.utils import top_k_top_p_filtering

import json
import random



def main():
    args = parse_args()

    # Set random reproducible seed
    torch.manual_seed(0)

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)

    past_key_values, input_ids, prompt_token_counts, position_ids, prompts = load_prompts_file(args, tokenizer)

    # Define 
    with torch.no_grad():
        # Now transition to running in token-by-token mode for generation
        if args.device in ['pytorch', 'golden', 'silicon'] and args.version != 'torch1.0':
            netlist_name = f'falcon_{"odlmh" if args.od_lm_head else ""}{"p" if args.padded_lm_head else ""}{"fv" if args.fracture_vocab else ""}_{"flash" if args.flash_decode else ""}_{args.num_chips}c_{args.fracture_mlp}mf_{args.fracture_attn}af_{args.num_layers}l_{args.seqlen}s_{args.net_name}'
            model.transformer.blocks = PyBudify(
                model.transformer.blocks,
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
                concurrent=(args.mode=='concurrent'),
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

        if args.version == 'efficient-40b':
            # assert args.mode == 'sequential', 'Only sequential mode supported for efficient-40b'
            model.transformer.blocks.to(torch.bfloat16)
            # model.lm_head.to(torch.float32)
            run_func = run_sync_efficient_40b
        elif args.version == 'tt' or args.version == 'split-mq':
            run_func = run_sync
        else:
            raise ValueError(f"Unknown version {args.version}")
        
        all_text = run_func(args, model, tokenizer, input_ids, past_key_values, prompt_token_counts, position_ids, prompts)

        if args.output_at_end:
            print(all_text)

def parse_args():
    parser = ArgumentParser('Generate text token-by-token starting with a pre-filled KV cache')
    parser.add_argument('-m', '--model', type=str, default='tiiuae/falcon-40b', help='Model name')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size (number of model inputs, not necessarily number of users, see --user-rows)')
    parser.add_argument('--user-rows', type=int, default=1, help='Number of users packed into the rows of each input (affects pybuda devices only)')
    parser.add_argument('--mode', choices=['sequential', 'concurrent', 'huggingface'], help='Concurrent continuously streams batch-size users through the model, huggingface runs the unmodified model')
    parser.add_argument('-s', '--stop', type=str, default='\n\n', help='Text to stop decoding after')
    parser.add_argument('-n', '--num-tokens', type=int, default=10, help='Maximum number of tokens to generate')
    parser.add_argument('-l', '--num-layers', type=int, default=60, help='Number of layers (default=all)')
    parser.add_argument('--output-at-end', action='store_true', help='Output at the end of generation instead of token by token')
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length')

    parser.add_argument('-d', '--device', choices=['cuda', 'huggingface', 'pytorch', 'golden', 'silicon'], default='huggingface', help='huggingface: run using HF code only, pytorch: use our shim but run in PyTorch, golden/silicon: run via pybuda')
    parser.add_argument('--arch', choices=['greyskull', 'wormhole', 'wormhole_b0', 'galaxy', 'nebula-galaxy'], default='wormhole_b0', help='Architecture to use for silicon')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16', 'fp8', 'fp8b'], default='fp32', help='Precision to use for all silicon tensors')
    parser.add_argument('--amp-level', type=int, choices=[0, 1, 2], help='Automatic mixed precision level (0=off, 1=mixed b-formats, 2=mixed a-formats)')
    parser.add_argument('--num-chips', type=int, default=1, help='Number of chips to use')
    parser.add_argument('--skip-lm-head', action='store_true', help='Skip the LM head and output garbage')
    parser.add_argument('--place', choices=['none', 'single', 'dual'], default='none', help='Manual placement of ops and buffers, default is none')
    parser.add_argument('--fuse', action='store_true', help='Fuse layers')
    parser.add_argument('--host-queues', action='store_true', help='Place input queues in host DRAM and let device read them over PCIe instead of copying them to device DRAM')
    parser.add_argument('--perf', choices=['none', 'light', 'verbose'], default=None, help='Performance tracing')
    parser.add_argument('--verify', action='store_true', help='Verify results')
    parser.add_argument('--log-level', choices=['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='ERROR', help='Log level')
    parser.add_argument('--load', type=str, help='Load a TTImage')
    parser.add_argument('--save', type=str, help='Save a TTImage')

    parser.add_argument('--version', type=str, choices=['odkv_t', 'odkv', 'tt', 'torch1.0', 'split-mq', 'efficient-40b','debug'], help='Version of the model to use')
    parser.add_argument('--flash-decode', action='store_true', help='Flash decode mode')

    parser.add_argument('-af', '--fracture-attn', type=int, default=0, help='Attention fracturing factor for version padded-fractured-full')
    parser.add_argument('-mf', '--fracture-mlp', type=int, default=0, help='MLP fracturing factor for version padded-fractured-full')
    parser.add_argument('--net-name', type=str, default='', help='Changes the netlist name')

    parser.add_argument('--load-pretrained', action='store_true', help='Load weights from hf.')
    # save and load weights because hf weights take way toooooo long to load
    parser.add_argument('--save-weights', action='store_true', help='Save weights to disk. Used for debugging.')
    parser.add_argument('--save-weights-dir', type=str, default='falcon40b.pt',help='Directory to save weights to. Used for debugging.')
    parser.add_argument('--load-weights', action='store_true', help='Load weights from disk. Used for debugging.')
    parser.add_argument('--load-weights-dir', type=str, default='falcon40b.pt',help='Directory to load weights from. Used for debugging.')

    parser.add_argument('--multi-chip-placer', type=str, choices=['parallel', 'pipeline'], default='parallel', help='Parallel or pipeline placer for multi-chip systems')
    parser.add_argument('--opt-level', type=int, default=0, help='Runtime optimization level')

    parser.add_argument('--enable-tvm-cache', action='store_true', help='Enable TVM caching')
    parser.add_argument('--hf-cache', type=str, default='/proj_sw/user_dev/hf_data', help='Cache directory for huggingface')

    parser.add_argument('--queues-on-host', action='store_true', help='Set flag os.environ["PYBUDA_ENABLE_OUTPUT_QUEUES_ON_HOST"] = "0"')
    parser.add_argument('-odlmh', '--od-lm-head', action='store_true', help='Put LM head on device')
    parser.add_argument('-plmh', '--padded-lm-head', action='store_true', help='Put padded LM head on device')
    parser.add_argument('-fv', '--fracture-vocab', action='store_true', help='Fracture vocab')
    parser.add_argument('--fracture-vocab-factor', type=int, default=8, help='Fracture vocab factor')
    parser.add_argument('--top-p', default=None, type=float, help='Top p sampling threshold (default: greedy)')
    parser.add_argument('--top-k', default=10, type=int, help='Top k sampling threshold (default: 10)')
    parser.add_argument('-t', '--temperature', default=1.0, type=float, help='Sampling temperature (default: 1.0)')
    parser.add_argument('--print-option', choices=['reprint', 'output-to-file'], default='reprint', help='Print all options')
    parser.add_argument('--print-interval', type=int, default=1, help='Print interval')
    parser.add_argument('--num-outer-loops', type=int, default=1, help='Experimental: insert outer loop which calls run_generate this many times')

    parser.add_argument('--prompts-file', type=str, default='data/multi_prompt.json')

    return parser.parse_args()


def load_model_and_tokenizer(args):
    
    # Check if shared cache folder exists
    if os.path.isdir(args.hf_cache):
        print("Using hf_cache folder: ", args.hf_cache)
    else:
        print("Cache folder not found. Reverting to default location")

    # Important: HF libraries must be set after changing the os.environ cache directory!
    from models.falcon40b.configuration_RW import RWConfig
    from models.falcon40b.debug import RWForCausalLM as RWForCausalLMTTdebug
    from models.falcon40b.modelling_RW_torch1 import RWForCausalLM as RWForCausalLMTorch1
    from models.falcon40b.tt_modeling_RW import RWForCausalLM as RWForCausalLMTT
    from models.falcon40b.tt_modeling_RW_odkv import RWForCausalLM as RWForCausalODKV
    from models.falcon40b.tt_modeling_RW_odkv_t import RWForCausalLM as RWForCausalODKV_T
    from transformers import AutoModelForCausalLM, AutoTokenizer, logging

    logging.set_verbosity_error()

    # Get model config using model name
    config = RWConfig.from_pretrained(args.model, cache_dir=args.hf_cache)
    config.n_layer = args.num_layers

    assert args.version is not None, "Please specify a version of the model to use"

    if args.version == 'split-mq':
        config.split_mq = True

    if args.version == 'efficient-40b':
        config.efficient_40b = True

    if args.flash_decode:
        assert args.version == 'efficient-40b', "Flash decode only supported for efficient-40b version"
        config.flash_decode = True

    if args.user_rows > 1:
        assert args.version == 'split-mq' or args.version == 'efficient-40b' or args.version == 'odkv' or args.version == 'odkv_t', "User rows only supported for split-mq/efficient-40b/odkv version"

    # Load model
    if args.device in ['huggingface', 'cuda']:
        model = AutoModelForCausalLM.from_pretrained(args.model, config=config, trust_remote_code=True, cache_dir=args.hf_cache)
    else:
        if args.version == 'tt' or args.version == 'split-mq':
            config.user_rows = args.user_rows
            if args.load_weights:
                # Might speed up model loading by skipping initial weight initialization
                model = RWForCausalLMTT(config)
                model.load_state_dict(torch.load(args.load_weights_dir), strict=False)
            elif args.save_weights:
                model = RWForCausalLMTT(config)
                torch.save(model.state_dict(), args.save_weights_dir)
            elif args.load_pretrained:
                model = RWForCausalLMTT.from_pretrained(args.model, config=config, cache_dir=args.hf_cache)
            else:
                print('WARNING: initializing model with random weights')
                model = RWForCausalLMTT(config)

            model.transformer.split_qkv_weights() # necessary to pre-split qkv
        elif args.version == 'efficient-40b':
            print("Using version efficient-40b")
            config.user_rows = args.user_rows
            config.padded_lmh = args.padded_lm_head
            config.fracture_vocab = args.fracture_vocab
            if args.load_weights:
                # Might speed up model loading by skipping initial weight initialization
                model = RWForCausalLMTT(config)
                model.load_state_dict(torch.load(args.load_weights_dir), strict=False)
            elif args.save_weights:
                model = RWForCausalLMTT(config)
                torch.save(model.state_dict(), args.save_weights_dir)
            elif args.load_pretrained:
                model = RWForCausalLMTT.from_pretrained(args.model, config=config, cache_dir=args.hf_cache)
            else:
                print('WARNING: initializing model with random weights')
                model = RWForCausalLMTT(config)
            
            model.transformer.split_qkv_wqkv_weights() # split wq weights for efficient-40b
            if args.padded_lm_head:
                model.set_padded_lmh(padding=512)
            if args.fracture_vocab:
                model.fracture_vocab(fracture_factor=args.fracture_vocab_factor)
            if args.od_lm_head:
                model.transformer.put_lm_head_on_device(model.lm_head)
        else:
            raise ValueError(f"Unknown version {args.version}")

        if args.skip_lm_head:
            # replace model.lm_head with a no-op module
            model.lm_head = torch.nn.Identity()

    # Set model into inference state
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.hf_cache)

    return model, tokenizer


def load_prompts_file(args, tokenizer):
    # Load prompts from json
    prompts = json.load( open(args.prompts_file) )
    # Setup padding token id
    tokenizer.pad_token_id = 0 #'[PAD]'     # set pad token for tokenizer
    # Encode the prompt
    tokenized = tokenizer(prompts, padding='max_length', return_tensors="pt", max_length=args.seqlen)

    cpu_input_ids = tokenized.input_ids.clone().squeeze()
    prompt_token_counts = torch.stack([sum(mask) for mask in tokenized.attention_mask])

    # cpu_position_ids begin as zeros
    cpu_position_ids = torch.zeros(1, args.user_rows, dtype=torch.long)

    ext_past_kv = []
    # past_key_values is a tuple of [key, value] tensors, one for each layer
    # copy into the right parts of the key, value tensors
    
    for _ in range(args.num_layers):

        past_key = torch.zeros((args.user_rows, 8, args.seqlen, 64), dtype=torch.bfloat16)
        past_value = torch.zeros((args.user_rows, 8, args.seqlen, 64), dtype=torch.bfloat16)

        if args.device == 'pytorch':
            past_key = past_key.to(torch.float32)
            past_value = past_value.to(torch.float32)

        ext_past_kv.append([past_key, past_value])

    past_key_values = tuple(ext_past_kv)

    return past_key_values, cpu_input_ids, prompt_token_counts, cpu_position_ids, prompts, 


def update_masks(attention_mask, position_ids):
    # attn_mask: 1, args.user_rows, args.seqlen
    attention_mask[0,torch.arange(attention_mask.size(1)),position_ids] = 1
    kv_mask_id = position_ids.clone().view(1,-1).to(torch.int32) # batch, num_users
    return attention_mask, kv_mask_id

def get_next_input(output_tokens, prompt_tokens, position_ids, prompt_lengths):
    '''
    This function either picks up the next prompt token or the current prediction,
    depending on whether we have completed prefill or not.

    output_tokens:      32, predictions (long)
    prompt_tokens:      32xseqlen padded prompts (long)
    position_ids:       1x32 (long) tensor of position_ids for the currently generated token
    prompt_lengths:     32, tensor of number of tokens in each prompt
    '''
    output_ids = torch.zeros(32, dtype=torch.long)
    for i in range(output_tokens.size(0)):
        # position_id = 0, prompt_length = 2
        next_input_idx = position_ids[0][i]+1
        if next_input_idx >= prompt_lengths[i]:
            output_ids[i] = output_tokens[i]
        else:
            output_ids[i] = prompt_tokens[i][next_input_idx]

    return output_ids


def run_sync(args, model, tokenizer, input_ids, past_key_values, num_tokens, position_ids, attn_mask, prompts):
    raise NotImplementedError("run_sync not implemented for this version")

def run_sync_efficient_40b(args, model, tokenizer, input_ids, past_key_values, prompt_token_counts, position_ids, prompts):
    if args.device == 'pytorch':
        # make sure blocks are float32
        model.transformer.blocks.bound_module.to(torch.float32)

    with reprint_output(output_type="list", initial_len=args.user_rows, interval=100) as out:
        mlo = None
        def initialize_multi_line_output():
            mlo = MultiLineOutput(args.user_rows, out)
            for i in range(args.user_rows):
                mlo.set_label(i, f'User {i+1:02d}: ')
            return mlo

        dest = 'cuda' if args.device == 'cuda' else 'cpu'
        model.to(dest)
        all_text = ''
        new_tokens = 0
        latencies = []
        start = time()
        overall_start = time()
        assert args.user_rows > 1, "User rows > 1 supported for efficient-40b"
        host_latencies = {'pre_processing': [], 
                        'main_forward_part(pre)': [], 
                        'main_forward_part(blocks)': [], 
                        'main_forward_part(post)': [], 
                        'final_forward_part': [],
                        'top_pk_time': [], 
                        'post_processing': []}
        args.profile_host_runtime = True  # do this for now. TODO: add it as an argument to the script

        #output_ids = input_ids.squeeze().unsqueeze(0).clone()
        output_hidden_states = None

        if args.user_rows > 1:
            new_past_key_values = []
            for (k,v) in past_key_values:
                # split the kv in each group into separate tensor
                # k, v are size [32 x 8 x 2048 x 64]
                layer_ks = []
                layer_vs = []
                for n in range(k.shape[1]):
                    # want it to be [1 x 32 x 2048 x 64]
                    layer_k = k[:, [n], :, :].transpose(0, 1).to(dest)
                    layer_ks.extend([layer_k])
                    layer_v = v[:, [n], :, :].transpose(0, 1).to(dest)
                    layer_vs.extend([layer_v])
                
                new_past_key_values.append(layer_ks+layer_vs)


            past_key_values = tuple(new_past_key_values)
        
        import cProfile, pstats
        pr = cProfile.Profile()
        loop_count = 0
        output_buffer = ['' for _ in range(args.user_rows)]

        # Set up attn_mask at the beginning. KV mask is generated each iteration
        attention_mask = torch.zeros((1, args.user_rows, args.seqlen), dtype=torch.int32)

        # Save the prompt tokens
        prompt_tokens = input_ids
        # Pick out the first column as the first set of input_ids
        input_ids = input_ids[:,[0]].clone()

        while True:
            if loop_count == 2:
                pr.enable()

            start_pre = time()
            if args.stop and args.stop in all_text:
                break
            
            if args.num_tokens and new_tokens >= args.num_tokens:
                break

            # since we padded all sequence to the longest sequence in the batch, we can share the same token index for rw mask and attention mask
            attention_mask, kv_mask_id = update_masks(attention_mask, position_ids)

            input_ids = input_ids.squeeze().view(args.batch_size, args.user_rows).to(dest)
            position_ids = position_ids.squeeze().view(args.batch_size, args.user_rows).to(dest)
            attention_mask = attention_mask.view(args.batch_size, args.user_rows, -1).to(dest)

            end_pre = time()      
            outputs = model.main_forward_part(input_ids, 
                            position_ids=position_ids, 
                            attention_mask=attention_mask, 
                            kv_mask_id=kv_mask_id, 
                            past_key_values=past_key_values,
                            profile_runtime=args.profile_host_runtime,
                            on_device_lm_head=args.od_lm_head
                            )
            if args.profile_host_runtime:
                host_latencies['main_forward_part(pre)'].append(outputs[-1]['pre'])
                host_latencies['main_forward_part(blocks)'].append(outputs[-1]['blocks'])
                host_latencies['main_forward_part(post)'].append(outputs[-1]['post'])
                outputs = outputs[:-1]  # last output is the host runtime dictionary, ignore that
            
            start_final_fwd = time()
            if args.od_lm_head:
                output = outputs[0] if not args.fracture_vocab else torch.cat(outputs[:args.fracture_vocab_factor], dim=-1)
                output = output.to('cpu').squeeze()
                past_key_values_cpy = outputs[1:] if not args.fracture_vocab else outputs[args.fracture_vocab_factor:]
            else:
                assert args.fracture_vocab is False, "Fractured vocab not supported for non-ODLMHead"
                outputs = model.final_forward_part(outputs)
                output = outputs.logits
                output = output.to('cpu').squeeze()
                past_key_values_cpy = outputs.past_key_values
            end_final_fwd = time()
            # take only up to vocab_size logits in case padding
            vocab_size = model.vocab_size
            output = output[..., :vocab_size]

            if args.device == 'pytorch':
                # update kv cache
                # past_key_values is [key1_l1, ... key8_l1, value1_l1, ... value8_l1, key1_l2, ...]
                # shape is already good: [1, 32, 2048, 64]
                # change it to [
                new_past_key_values = []
                for i in range(args.num_layers):
                    new_past_key_values.append(past_key_values_cpy[i*16:(i+1)*16])
                past_key_values = tuple(new_past_key_values)

            toppk_time = time()
            if not args.top_p: # greedy
                token = output.argmax(dim=-1)
            else:
                token = top_pk_logits_efficient(output.to(torch.float32), args.top_p, args.top_k, args.temperature)
            toppk_time = time() - toppk_time

            # Choose between prefill or decode tokens
            token = get_next_input(token, prompt_tokens, position_ids, prompt_token_counts)

            input_ids = token.squeeze()
            #output_ids = torch.cat((output_ids, token.unsqueeze(0)), dim=1)

            # Update position ids for each user
            position_ids = position_ids + 1
            new_tokens += 1

            # Print the generated token
            if not args.output_at_end:
                text = tokenizer.decode(token, clean_up_tokenization_spaces=True)
            if args.print_option == 'reprint':
                if mlo is None:
                    print('Ready to begin')
                    input()
                    mlo = initialize_multi_line_output()
                    for i in range(args.user_rows):
                        text = prompts[i]
                        text = text.replace('\n', ' ')
                        mlo.append_line(i, text)

                for i in range(args.user_rows):
                    text = tokenizer.decode(token[i], clean_up_tokenization_spaces=True, skip_special_tokens=True)
                    text = text.replace('\n', ' ')
                    mlo.append_line(i, text)
                if loop_count % args.print_interval == 0:
                    mlo.refresh()
            elif args.print_option == 'output-to-file':
                if mlo is None:
                    mlo = True
                    with open(f'{args.device}_{args.version}_{args.num_layers}L_topp{args.top_p}_topk{args.top_k}_v0.txt', 'w') as f:
                        f.write(f'Ready to begin\n')
                    print('Ready to begin')
                    input()
                    with open(f'{args.device}_{args.version}_{args.num_layers}L_topp{args.top_p}_topk{args.top_k}_v0.txt', 'w') as f:
                        f.write(f'Ready to begin\n')
                        # Save the prompts
                        for i in range(args.user_rows):
                            prompts[i] = prompts[i].replace('\n', ' ')
                            f.write(f'User {i}: {prompts[i]}\n')
                else:
                    if loop_count % args.print_interval == 0:
                        # Update the file with new token at the end of each line
                        with open(f'{args.device}_{args.version}_{args.num_layers}L_topp{args.top_p}_topk{args.top_k}_v0.txt', 'w') as f:
                            f.write(f'Ready to begin\n')
                            for i in range(args.user_rows):
                                text = tokenizer.decode(token[i], clean_up_tokenization_spaces=True, skip_special_tokens=True)
                                text = text.replace('\n', ' ')
                                #prompts[i] += " "
                                prompts[i] += text
                                f.write(f'User {i}: {prompts[i]}\n')
                    else:
                        # just update the tokens in prompts
                        for i in range(args.user_rows):
                            text = tokenizer.decode(token[i], clean_up_tokenization_spaces=True, skip_special_tokens=True)
                            text = text.replace('\n', ' ')
                            #prompts[i] += " "
                            prompts[i] += text

            else:
                raise ValueError(f"Unknown print option {args.print_option}")
            

            end_post = time()

            latency = time() - start
            latencies.append(latency)
            start = time()

            if args.profile_host_runtime:
                host_latencies['pre_processing'].append(end_pre-start_pre)
                host_latencies['final_forward_part'].append(end_final_fwd - start_final_fwd)
                host_latencies['top_pk_time'].append(toppk_time)
                host_latencies['post_processing'].append(end_post - end_final_fwd)

            loop_count += 1
        pr.disable()
        sortby = 'cumulative'
        ps = pstats.Stats(pr).sort_stats(sortby)
        ps.dump_stats('stats.dmp')
        
        overall_time = time() - overall_start
        overall_tokens = args.batch_size * args.user_rows * args.num_tokens

        # save hidden states outputs
        # torch.save(output_hidden_states, f'{args.device}_{args.num_layers}L_hidden_states.pt')

        # torch.save((input_ids, past_key_values), f'{args.device}_{args.version}_{args.num_layers}L_kv.pt')

        warmup_batch = 2
        # skip initial warmup batch
        if len(latencies) > warmup_batch:
            overall_time -= sum(latencies[:warmup_batch])
            overall_tokens -= args.batch_size * args.user_rows
            latencies = latencies[warmup_batch:]

        #all_text = tokenizer.decode(output_ids[0], clean_up_tokenization_spaces=True)

        mean_latency = sum(latencies) / len(latencies)
        #print(f'All text:')
        #print(all_text)
        sleep(5)
        print(f'User latency: {1000 * mean_latency:.1f} ms @ {1/mean_latency:.1f} tokens/s')
        print(f'Overall throughput: {1000 * overall_time / overall_tokens:.1f} ms @ {overall_tokens / overall_time:.1f} tokens/s')
        if args.profile_host_runtime:
            print(f'Host latencies:')
            # print mean latencies for each host function, skupping the first warmup one
            for k, v in host_latencies.items():
                if len(v) > warmup_batch:
                    print(f'{k}: {1000 * sum(v[warmup_batch:]) / len(v[warmup_batch:]):.1f} ms')

        # save demo result to file
        write_type = 'a' if args.print_option == 'output-to-file' else 'w'
        with open(f'{args.device}_{args.version}_{args.num_layers}L_topp{args.top_p}_topk{args.top_k}_v0.txt', write_type) as f:
            f.write(f'User latency: {1000 * mean_latency:.1f} ms @ {1/mean_latency:.1f} tokens/s\n')
            f.write(f'Overall throughput: {1000 * overall_time / overall_tokens:.1f} ms @ {overall_tokens / overall_time:.1f} tokens/s\n')
            if args.profile_host_runtime:
                f.write(f'Host latencies:\n')
                # print mean latencies for each host function, skupping the first warmup one
                for k, v in host_latencies.items():
                    if len(v) > warmup_batch:
                        f.write(f'{k}: {1000 * sum(v[warmup_batch:]) / len(v[warmup_batch:]):.1f} ms\n')
            #f.write(f'All text:\n')
            #f.write(all_text)

        if args.device != 'pytorch':
            model.transformer.blocks.shutdown()
        return all_text

def pop_outputs(model):
        ys = model.transformer.blocks.output_q.get()
        outputs = tuple([ y.value().float() for y in ys ])
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

def top_pk_logits(logits, p=0.9, k=10, temperature=1.0, return_probs=False):
    next_token_logscores = top_k_top_p_filtering(logits, top_k=k, top_p=p)
    probs = F.softmax(next_token_logscores/temperature, dim=-1)
    token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    if return_probs:
        return token, probs
    else:
        return token

def top_pk_logits_efficient(logits, p=0.9, k=10, temperature=1.0, return_probs=False):
    # do not keep the entire vocab size after top k. Instead, keep the k size tensor and record the associated indices
    top_k_values, top_k_indices = torch.topk(logits, k=k)
    top_p_values = top_k_top_p_filtering(top_k_values, top_p=p)
    probs = F.softmax(top_p_values/temperature, dim=-1)
    top_k_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token = top_k_indices.gather(-1, top_k_id.unsqueeze(-1)).squeeze(-1)
    if return_probs:
        return token, (probs, top_k_indices)
    else:
        return token

if __name__ == '__main__':
    main()
