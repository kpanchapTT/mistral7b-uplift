import os
import queue
import sys
import time

import torch
from DecodeConfig import DecodeConfig
from PrefillConfig import PrefillConfig


class PyBudify(torch.nn.Module):
    def __init__(
        self,
        pt_module,
        device="silicon",
        arch="wormhole_b0",
        precision="fp32",
        amp_level=0,
        micro_batch_size=1,
        fuse=False,
        num_chips=1,
        perf=None,
        verify=False,
        log_level="ERROR",
        tti_save=None,
        tti_load=None,
        concurrent=False,
        netlist_name="pybudify_module",
        version="tt",
        is_decode=True,
        opt_level=0,
        fracture_mlp=0,
        enable_tvm_cache=False,
        user_rows=1,
        num_tokens=1,
        queues_on_host=False,
        od_lm_head=False,
        fracture_vocab=False,
        fracture_vocab_factor=8,
        flash_decode=False,
        num_outer_loops=1,
        **kwargs,
    ):
        super().__init__()

        self.device = device
        self.bound_module = pt_module
        self.tti_save = tti_save
        self.tti_load = tti_load
        self.concurrent = concurrent
        self.version = version  #
        self.num_chips = num_chips
        self.num_tokens = num_tokens  # ???
        self.queues_on_host = queues_on_host
        self.arch = arch
        self.flash_decode = flash_decode
        self.num_outer_loops = num_outer_loops

        if device == "pytorch":
            return

        if log_level:
            os.environ["LOGGER_LEVEL"] = log_level
            os.environ["LOGURU_LEVEL"] = log_level

        pybuda = self.pybuda = __import__(
            "pybuda"
        )  # let us set log levels before importing pybuda

        if enable_tvm_cache:
            set_tvm_cache(pybuda, netlist_name)

        devtype = {
            "golden": pybuda.BackendType.Golden,
            "silicon": pybuda.BackendType.Silicon,
        }[device]

        module = pybuda.PyTorchModule(netlist_name, self.bound_module)

        if precision == "fp32":
            fallback = pybuda.DataFormat.Float32
        elif precision == "fp16":
            fallback = pybuda.DataFormat.Float16
        elif precision == "bf16":
            fallback = pybuda.DataFormat.Float16_b
        elif precision == "fp8":
            fallback = pybuda.DataFormat.Bfp8
        elif precision == "fp8b":
            fallback = pybuda.DataFormat.Bfp8_b
        else:
            raise ValueError('Precision "%s" not implemented' % precision)

        perf_level = {
            None: None,
            "none": None,
            "light": pybuda.PerfTraceLevel.LIGHT,
            "verbose": pybuda.PerfTraceLevel.VERBOSE,
        }[perf]

        if is_decode:
            decode_config = DecodeConfig(
                pybuda=pybuda,
                pt_module=pt_module,
                default_df_override=fallback,
                accumulate_df=pybuda.DataFormat.Float32,
                amp_level=amp_level,
                enable_auto_fusing=fuse,
                performance_trace=perf_level,
                backend_opt_level=opt_level,
                enable_auto_transposing_placement=True,
                enable_t_streaming=True,
                manual_t_streaming=True,
                version=version,
                flash_decode=flash_decode,
                fracture_mlp=fracture_mlp,
                queues_on_host=queues_on_host,
                arch=arch,
                chip_ids=list(range(num_chips)),
            )
            # Override placements
            decode_config.placement_overrides(
                fracture_vocab=fracture_vocab,
                fracture_vocab_factor=fracture_vocab_factor,
                num_chips=num_chips,
                user_rows=user_rows,
                od_lm_head=od_lm_head,
            )
            chip_ids = decode_config.get_chip_ids()
        else:
            # prefill
            prefill_config = PrefillConfig(
                pybuda=pybuda,
                pt_module=pt_module,
                version=version,
                fracture_mlp=fracture_mlp,
                default_df_override=fallback,
                accumulate_df=pybuda.DataFormat.Float32,
                amp_level=amp_level,
                enable_auto_fusing=fuse,
                performance_trace=perf_level,
                backend_opt_level=opt_level,
                enable_auto_transposing_placement=True,
                enable_t_streaming=True,
                manual_t_streaming=True,
                queues_on_host=queues_on_host,
                arch=arch,
                chip_ids=list(range(num_chips)),
            )
            # Override placements
            prefill_config.placement_overrides()

            chip_ids = prefill_config.get_chip_ids()

        pybuda_arch = {
            "grayskull": pybuda.BackendDevice.Grayskull,
            "wormhole": pybuda.BackendDevice.Wormhole,
            "wormhole_b0": pybuda.BackendDevice.Wormhole_B0,
            "galaxy": pybuda.BackendDevice.Wormhole_B0,
            "nebula-galaxy": pybuda.BackendDevice.Wormhole_B0,
        }[arch]

        if tti_load is not None:
            print(f"Loading image from {tti_load}")
            loading_start_time = time.time()
            self.tt0 = pybuda.TTDevice.load_image(img_path=self.tti_load)
            loading_end_time = time.time()
            print(
                f"Model Loading Finished. Successfully Initialized in {loading_end_time-loading_start_time} seconds."
            )
        else:
            self.tt0 = pybuda.TTDevice(
                "tt0",
                module=module,
                fp32_fallback=fallback,
                arch=pybuda_arch,
                devtype=devtype,
                chip_ids=chip_ids,
            )

        # mp = torch.multiprocessing.get_context('spawn')
        # self.output_q = mp.Queue()
        # We think mp queues were causing a shmem issue for sequential mode
        self.output_q = queue.Queue()
        if self.concurrent == False:
            os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"

        if verify:
            self.verify_cfg = pybuda.VerifyConfig(
                verify_all=True,
                verify_last=True,
                devtype=pybuda.BackendType.Silicon,
                arch=pybuda_arch,
            )
        else:
            self.verify_cfg = None

        self.initialized = False
        self.micro_batch_size = micro_batch_size

    def ensure_initialized(self, *args):
        if not self.initialized and self.device != "pytorch":
            compile_start_time = time.time()
            self.num_generated = 0
            if self.tti_save is not None:
                self.tt0.compile_to_image(
                    img_path=self.tti_save,
                    training=False,
                    sample_inputs=args,
                    microbatch_count=self.micro_batch_size,
                )
                print(f"Saved image to {self.tti_save}")
                self.pybuda.config._clear_global_compiler_config()
                self.pybuda.pybuda_reset()
                self.pybuda.shutdown()
                sys.exit(0)
            # breakpoint()
            self.pybuda.initialize_pipeline(
                training=False,
                sample_inputs=args,
                output_queue=self.output_q,
                microbatch_count=self.micro_batch_size,
                _sequential=self.concurrent == False,
                _verify_cfg=self.verify_cfg,
            )
            compile_end_time = time.time()
            print(
                f"Model Compilation Finished. Successfully Initialized in {compile_end_time-compile_start_time} seconds."
            )
            if self.concurrent:
                if self.version == "efficient-40b":
                    pass
                    # self.pybuda.run_generate(input_count=self.num_tokens//self.num_outer_loops, write_index=0)
                else:
                    self.pybuda.run_forward(input_count=self.num_tokens)
        self.initialized = True

    def __call__(self, *args, **kwargs):
        if self.device == "pytorch":
            result = self.bound_module(*args, **kwargs)
            return result
        else:
            self.ensure_initialized(*args)

            if self.version == "efficient-40b":
                if not self.concurrent:
                    self.pybuda.sync()
                else:
                    if (
                        self.num_generated % (self.num_tokens // self.num_outer_loops)
                        == 0
                    ):
                        self.pybuda.run_generate(
                            input_count=self.num_tokens // self.num_outer_loops,
                            write_index=0,
                        )
                    self.num_generated += 1

                if self.version == "efficient-40b":
                    self.tt0.push_to_inputs(args[:4] + args[-1:])
                else:
                    self.tt0.push_to_inputs(
                        args[:6]
                    )  # don't pass in kv over and over again

                if not self.concurrent:
                    self.pybuda.run_generate(
                        input_count=1, write_index=0, _sequential=True
                    )

                ys = self.output_q.get()
                # outputs = tuple([ y.value().float() for y in ys if isinstance(y, self.pybuda.tensor.TensorFromPytorch)])
                outputs = tuple(
                    [
                        y.value()
                        for y in ys
                        if isinstance(y, self.pybuda.tensor.TensorFromPytorch)
                    ]
                )
                if len(outputs) == 1:
                    outputs = outputs[0]
                if self.verify_cfg:
                    baseline = self.bound_module(*args, **kwargs)
                    if len(outputs) != len(baseline):
                        print(f"Num outputs: {len(outputs)}, expected: {len(baseline)}")
                    for i, (real, expected) in enumerate(zip(outputs, baseline)):
                        pcc = torch.corrcoef(
                            torch.stack([real.reshape(-1), expected.reshape(-1)])
                        )[0, 1]
                        print("PCC tensor %d: %.4f" % (i, pcc))

                # import pdb; pdb.set_trace()

                result = outputs

                return result
            else:
                self.tt0.push_to_inputs(*args)
                if not self.concurrent:
                    self.pybuda.run_forward(input_count=1, _sequential=True)
                ys = self.output_q.get()
                # outputs = tuple([ y.value().float() for y in ys if isinstance(y, self.pybuda.tensor.TensorFromPytorch)])
                outputs = tuple(
                    [
                        y.value()
                        for y in ys
                        if isinstance(y, self.pybuda.tensor.TensorFromPytorch)
                    ]
                )
                if len(outputs) == 1:
                    outputs = outputs[0]

                return outputs

    def shutdown(self):
        self.pybuda.shutdown()


def set_tvm_cache(pybuda, netlist_name):
    os.environ["PYBUDA_ENABLE_TVM_CACHE"] = "1"
    global_config = pybuda.config._get_global_compiler_config()

    # os current working directory
    tvm_path = os.getcwd() + "/tvm_" + netlist_name
    print("Pointing TVM cache to ", tvm_path)
    # if tvm path exists, load from it
    if os.path.exists(tvm_path):
        global_config.tvm_graph_load_path = tvm_path
    # else, save to it
    else:
        global_config.tvm_graph_store_path = tvm_path
