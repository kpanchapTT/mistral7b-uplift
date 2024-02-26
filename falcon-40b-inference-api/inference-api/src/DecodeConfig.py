import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DecodeConfig:
    pybuda: Any
    pt_module: Any
    default_df_override: Any
    accumulate_df: Any
    amp_level: int = 0
    enable_auto_fusing: bool = False
    performance_trace: Any = None
    backend_opt_level: int = 0
    enable_auto_transposing_placement: bool = True
    enable_t_streaming: bool = True
    manual_t_streaming: bool = True

    version: str = None
    flash_decode: bool = False
    fracture_mlp: int = 0
    queues_on_host: bool = False
    arch: str = None
    chip_ids: list = field(default_factory=list)

    def __post_init__(self):
        self.env_var_setup()

    def env_var_setup(self):
        os.environ["GOLDEN_WORMHOLE_B0"] = "1"
        os.environ["PYBUDA_ENABLE_BROADCAST_SPLITTING"] = "1"
        os.environ["PYBUDA_ENABLE_STABLE_SOFTMAX"] = "1"
        os.environ["PYBUDA_CONVERT_PARAMS_TO_TVM"] = "0"
        os.environ["TT_BACKEND_TIMEOUT"] = "0"
        os.environ["TT_BACKEND_POP_TIMEOUT"] = "10000"
        os.environ["TT_BACKEND_PUSH_TIMEOUT"] = "10000"
        os.environ["TT_BACKEND_GET_TIMEOUT"] = "10000"

        os.environ["PYBUDA_ENABLE_OUTPUT_QUEUES_ON_HOST"] = "1"
        os.environ["PYBUDA_DISABLE_DYNAMIC_DRAM"] = "1"
        # we want to force thread
        os.environ["PYBUDA_FORCE_THREADS"] = "1"
        os.environ["TT_BACKEND_EPOCH_BIN_NUM_SLOTS"] = "128"
        os.environ["PYBUDA_DISABLE_DRAM0"] = "1"

        if self.queues_on_host:
            os.environ["PYBUDA_ENABLE_OUTPUT_QUEUES_ON_HOST"] = "0"

        if self.arch == "nebula-galaxy":
            os.environ["TT_BACKEND_HETEROGENOUS_HARVESTING"] = "0"

    def placement_overrides(
        self, fracture_vocab, fracture_vocab_factor, num_chips, user_rows, od_lm_head
    ):
        pybuda = self.pybuda
        pt_module = self.pt_module
        # Required or we get invalid DF error.
        # Important: DO not set intermed, acc_df or we hang on prefill.
        pybuda.config.configure_mixed_precision(
            op_type="splice",
            output_df=pybuda.DataFormat.Float16_b,
            input_df={
                0: [pybuda.DataFormat.Float16_b, True],
                1: [pybuda.DataFormat.Float16_b, True],
                2: [pybuda.DataFormat.Float16_b, True],
            },
        )

        pybuda.config.configure_mixed_precision(
            name_regex="sparse_matmul",
            output_df=pybuda.DataFormat.Float16_b,
            intermediate_df=pybuda.DataFormat.Float16_b,
        )

        # embeddig on device
        pybuda.config._get_global_compiler_config().cpu_fallback_ops.remove("embedding")

        pybuda.set_configuration_options(
            default_df_override=self.default_df_override,
            accumulate_df=self.accumulate_df,
            amp_level=self.amp_level,
            enable_auto_fusing=self.enable_auto_fusing,
            performance_trace=self.performance_trace,
            backend_opt_level=self.backend_opt_level,
            enable_auto_transposing_placement=self.enable_auto_transposing_placement,
            enable_t_streaming=self.enable_t_streaming,
            manual_t_streaming=self.manual_t_streaming,
        )

        if self.version == "efficient-40b":
            name_base_kv = [1, 3, 5, 7, 9, 11, 13, 15]
            names = []
            for i in range(len(self.pt_module.layers)):
                names.extend(
                    [f"k_{name_base_kv[j]+i*48+16}" for j in range(8)]
                    + [f"v_{name_base_kv[j]+i*32}" for j in range(8)]
                )

            print(f'names" {names}')
            if fracture_vocab:
                names_dict = {
                    name: (i + fracture_vocab_factor) for i, name in enumerate(names)
                }
            else:
                names_dict = {name: (i + 1) for i, name in enumerate(names)}

            compiler_cfg = pybuda.config._get_global_compiler_config()
            compiler_cfg.loopback_outputs = names_dict

        if num_chips > 1:
            if self.version == "efficient-40b":
                print("Using efficient-40b")
                kv_update_sched = (
                    []
                )  # process every 3 layers together (flash decode only)

                if (
                    num_chips == 32
                    and user_rows == 32
                    and self.fracture_mlp == 8
                    and self.flash_decode
                ):
                    print(
                        f"32 chips {len(pt_module.layers)} layer 32 user rows {self.fracture_mlp} mlp fractures -- mlp fractured -- flash_decode"
                    )
                    # os.environ['TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE'] = "49152"
                    # chip_ids = [0,29,27,28,31,5,4,30,3,1,25,24,23,22,21,13,14,20,19,15,16,17,18,2,26,6,7,8,9,10,11,12]
                    if self.arch == "nebula-galaxy":
                        print("Using galaxy nebula chip orders:")
                        # chip_ids = [0, 3,17,16,11,12,10,13,18,15, 1, 4,27,28,29,30,31,32,25,26, 6,24,23,22,21,20,19,7, 8, 9,14,5, 2] # 201
                        chip_ids = [
                            0,
                            20,
                            23,
                            22,
                            13,
                            14,
                            12,
                            15,
                            24,
                            21,
                            26,
                            19,
                            2,
                            1,
                            31,
                            32,
                            29,
                            30,
                            4,
                            3,
                            18,
                            5,
                            6,
                            28,
                            27,
                            7,
                            8,
                            9,
                            10,
                            11,
                            16,
                            17,
                            25,
                        ]  # aus-glx-1, aus-glx-02, aus-glx-03
                        # chip_ids = [0, 21, 1, 4, 6, 5, 7, 20, 2, 3, 31, 22, 30, 29, 28, 27, 26, 25, 24, 23, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 19, 18, 32] # aus-glx-01 (old)
                        print(chip_ids)
                    else:
                        chip_ids = [
                            0,
                            3,
                            31,
                            28,
                            26,
                            27,
                            6,
                            5,
                            30,
                            29,
                            1,
                            2,
                            25,
                            24,
                            23,
                            22,
                            21,
                            20,
                            19,
                            18,
                            16,
                            15,
                            14,
                            13,
                            12,
                            11,
                            10,
                            9,
                            8,
                            7,
                            4,
                            17,
                        ]  # 206
                    # we start placing other (not mlp) ops from mlp_factor+2 chip

                    ######## manual overrides
                    offset_base = 622
                    end_offset = len(pt_module.layers) * offset_base
                    subtract_id = end_offset + 29
                    daisy_cache = [
                        "sin",
                        "cos",
                        f"subtract_{subtract_id+1}",
                        f"subtract_{subtract_id}",
                        "mask",
                    ]  # want daisy chain to wrap around for each layer

                    # make ensure masks saved into mmio chip 0 (if not default placed there)
                    pybuda.config.override_dram_queue_placement("mask", chip_id=0)
                    pybuda.config.override_dram_queue_placement("input_1", chip_id=0)

                    def insert_by_groups(
                        input,
                        groups,
                        verbose=False,
                        return_intermediates=False,
                        override_size=None,
                    ):
                        """
                        takes in [g1, g2, g3...] and g1 = [e1, e2, e3...]
                        input -- g1 -- g2 -- g3
                        """
                        intermediates = []
                        for i in range(len(groups)):
                            sub_group = [
                                item for sublist in groups[i:] for item in sublist
                            ]  # flatten groups
                            if verbose:
                                print(
                                    "[self.insert_nops_by_groups]:",
                                    f"input: {input}",
                                    f"sub_group: {sub_group}",
                                )
                            input = self.insert_nop(input, sub_group)
                            if override_size is not None:
                                pybuda.override_op_size(input, override_size)
                            intermediates.append(input)
                        if return_intermediates:
                            return intermediates
                        else:
                            return input

                    # cross layer daisy chains
                    sin_ids = [
                        20,
                        39,
                        95,
                        114,
                        171,
                        190,
                        246,
                        265,
                        323,
                        342,
                        398,
                        417,
                        474,
                        493,
                        549,
                        568,
                    ]  # [20, 39, 92, 111, 164, 183, 236, 255, 308, 327, 380, 399, 452, 471, 524, 543] #[20,39,91,110,162,181,233,252,304,323,375,394,446,465,517,536]
                    cos_ids = [
                        15,
                        34,
                        90,
                        109,
                        166,
                        185,
                        241,
                        260,
                        318,
                        337,
                        393,
                        412,
                        469,
                        488,
                        544,
                        563,
                    ]  # [15, 34, 87, 106, 159, 178, 231, 250, 303, 322, 375, 394, 447, 466, 519, 538] #[15,34,86,105,157,176,228,247,299,318,370,389,441,460,512,531]
                    k_ids = [
                        31 + end_offset,
                        34 + end_offset,
                        37 + end_offset,
                        40 + end_offset,
                        43 + end_offset,
                        46 + end_offset,
                        49 + end_offset,
                        52 + end_offset,
                    ]  # k+end_offset for k in [31, 35, 39, 43, 47, 51, 55, 59]] #[
                    v_ids = [
                        55 + end_offset,
                        58 + end_offset,
                        61 + end_offset,
                        64 + end_offset,
                        67 + end_offset,
                        70 + end_offset,
                        73 + end_offset,
                        76 + end_offset,
                    ]  # [v+end_offset for v in [55, 59, 63, 67, 71, 75, 79, 83]] #
                    att_add_ids = [
                        29,
                        104,
                        180,
                        255,
                        332,
                        407,
                        483,
                        558,
                    ]  # [29, 101, 173, 245, 317, 389, 461, 533] # [29, 100, 171, 242, 313, 384, 455, 526]
                    for i in range(8):
                        sg, cg, rg_k, rg_v, wg_k, wg_v, aag = [], [], [], [], [], [], []
                        for layer in range(len(pt_module.layers)):
                            rw_ids = [
                                [l, l + 8]
                                for l in range(layer * 16, (layer + 1) * 16 - 8)
                            ]
                            offset = layer * offset_base
                            offset_2 = layer * 48
                            sg.extend(
                                [
                                    f"multiply_{sin_ids[2*i]+offset}",
                                    f"multiply_{sin_ids[2*i+1]+offset}",
                                ]
                            )
                            cg.extend(
                                [
                                    f"multiply_{cos_ids[2*i]+offset}",
                                    f"multiply_{cos_ids[2*i+1]+offset}",
                                ]
                            )
                            rg_k.extend([f"multiply_{k_ids[i]+1+offset_2}"])
                            wg_k.extend([f"multiply_{k_ids[i]+offset_2}"])
                            rg_v.extend([f"multiply_{v_ids[i]+1+offset_2}"])
                            wg_v.extend([f"multiply_{v_ids[i]+offset_2}"])

                            aag.extend([f"add_{att_add_ids[i]+offset}"])

                        # Insert daisy chain for each MQA going down the layers
                        pybuda.config.insert_buffering_nop(
                            daisy_cache[0], sg, hoist_tms=False, daisy_chain=True
                        )
                        print("Inserting Nops between", daisy_cache[0], sg)

                        pybuda.config.insert_buffering_nop(
                            daisy_cache[1], cg, hoist_tms=False, daisy_chain=True
                        )
                        print("Inserting Nops between", daisy_cache[1], cg)

                        pybuda.config.insert_buffering_nop(
                            daisy_cache[3], rg_k, daisy_chain=True
                        )
                        print("Inserting Nops between", daisy_cache[3], rg_k)

                        pybuda.config.insert_buffering_nop(
                            daisy_cache[3], rg_v, daisy_chain=True
                        )
                        print("Inserting Nops between", daisy_cache[3], rg_v)

                        pybuda.config.insert_buffering_nop(
                            daisy_cache[2], wg_k, daisy_chain=True
                        )
                        print("Inserting Nops between", daisy_cache[2], wg_k)

                        pybuda.config.insert_buffering_nop(
                            daisy_cache[2], wg_v, daisy_chain=True
                        )
                        print("Inserting Nops between", daisy_cache[2], wg_v)

                        pybuda.config.insert_buffering_nop(
                            daisy_cache[4], aag, daisy_chain=True
                        )
                        print("Inserting Nops between", daisy_cache[4], aag)

                        [
                            pybuda.config.override_op_size(
                                f"buffer_0_{daisy_cache[4]}_{attn_op}", (1, 4)
                            )
                            for attn_op in aag
                        ]

                    # Inserting 2D daisy chain for cos & sin leads to a strange error in `Lowering to Buda`. The error is that eltwise input must be broadcastable.
                    first_sin_group = [f"multiply_{sin_ids[2*i]}" for i in range(8)]
                    first_sin_group_bufs = [
                        f"buffer_0_{daisy_cache[0]}_{sin_op}"
                        for sin_op in first_sin_group
                    ]
                    print(
                        f"inserting daisy_chain nops between {daisy_cache[0]} and {first_sin_group_bufs}"
                    )
                    pybuda.config.insert_buffering_nop(
                        daisy_cache[0],
                        first_sin_group_bufs,
                        hoist_tms=False,
                        daisy_chain=True,
                    )
                    first_sin_group_bufs = [
                        f"buffer_0_{daisy_cache[0]}_{sin_op}"
                        for sin_op in first_sin_group_bufs
                    ]

                    first_cos_group = [f"multiply_{cos_ids[2*i]}" for i in range(8)]
                    first_cos_group_bufs = [
                        f"buffer_0_{daisy_cache[1]}_{cos_op}"
                        for cos_op in first_cos_group
                    ]
                    print(
                        f"inserting daisy_chain nops between {daisy_cache[1]} and {first_cos_group_bufs}"
                    )
                    pybuda.config.insert_buffering_nop(
                        daisy_cache[1],
                        first_cos_group_bufs,
                        hoist_tms=False,
                        daisy_chain=True,
                    )
                    first_cos_group_bufs = [
                        f"buffer_0_{daisy_cache[1]}_{cos_op}"
                        for cos_op in first_cos_group_bufs
                    ]

                    first_read_group_k = [f"multiply_{k_ids[i]+1}" for i in range(8)]
                    first_read_group_k_bufs = [
                        f"buffer_0_{daisy_cache[3]}_{read_op}"
                        for read_op in first_read_group_k
                    ]
                    print(
                        f"inserting daisy_chain nops between {daisy_cache[3]} and {first_read_group_k_bufs}"
                    )
                    pybuda.config.insert_buffering_nop(
                        daisy_cache[3], first_read_group_k_bufs, daisy_chain=True
                    )
                    [
                        pybuda.config.override_op_size(
                            f"buffer_0_{daisy_cache[3]}_buffer_0_{daisy_cache[3]}_{read_op}",
                            (1, 2),
                        )
                        for read_op in first_read_group_k
                    ]
                    first_read_group_k_bufs = [
                        f"buffer_0_{daisy_cache[3]}_{read_op}"
                        for read_op in first_read_group_k_bufs
                    ]

                    first_read_group_v = [f"multiply_{v_ids[i]+1}" for i in range(8)]
                    first_read_group_v_bufs = [
                        f"buffer_0_{daisy_cache[3]}_{read_op}"
                        for read_op in first_read_group_v
                    ]
                    print(
                        f"inserting daisy_chain nops between {daisy_cache[3]} and {first_read_group_v_bufs}"
                    )
                    pybuda.config.insert_buffering_nop(
                        daisy_cache[3], first_read_group_v_bufs, daisy_chain=True
                    )
                    [
                        pybuda.config.override_op_size(
                            f"buffer_0_{daisy_cache[3]}_buffer_0_{daisy_cache[3]}_{read_op}",
                            (1, 2),
                        )
                        for read_op in first_read_group_v
                    ]
                    first_read_group_v_bufs = [
                        f"buffer_0_{daisy_cache[3]}_{read_op}"
                        for read_op in first_read_group_v_bufs
                    ]

                    first_write_group_k = [f"multiply_{k_ids[i]}" for i in range(8)]
                    first_write_group_k_bufs = [
                        f"buffer_0_{daisy_cache[2]}_{write_op}"
                        for write_op in first_write_group_k
                    ]
                    print(
                        f"inserting daisy_chain nops between {daisy_cache[2]} and {first_write_group_k_bufs}"
                    )
                    pybuda.config.insert_buffering_nop(
                        daisy_cache[2], first_write_group_k_bufs, daisy_chain=True
                    )
                    [
                        pybuda.config.override_op_size(
                            f"buffer_0_{daisy_cache[2]}_buffer_0_{daisy_cache[2]}_{write_op}",
                            (1, 2),
                        )
                        for write_op in first_write_group_k
                    ]
                    first_write_group_k_bufs = [
                        f"buffer_0_{daisy_cache[2]}_{write_op}"
                        for write_op in first_write_group_k_bufs
                    ]

                    first_write_group_v = [f"multiply_{v_ids[i]}" for i in range(8)]
                    first_write_group_v_bufs = [
                        f"buffer_0_{daisy_cache[2]}_{write_op}"
                        for write_op in first_write_group_v
                    ]
                    print(
                        f"inserting daisy_chain nops between {daisy_cache[2]} and {first_write_group_v_bufs}"
                    )
                    pybuda.config.insert_buffering_nop(
                        daisy_cache[2], first_write_group_v_bufs, daisy_chain=True
                    )
                    [
                        pybuda.config.override_op_size(
                            f"buffer_0_{daisy_cache[2]}_buffer_0_{daisy_cache[2]}_{write_op}",
                            (1, 2),
                        )
                        for write_op in first_write_group_v
                    ]
                    first_write_group_v_bufs = [
                        f"buffer_0_{daisy_cache[2]}_{write_op}"
                        for write_op in first_write_group_v_bufs
                    ]

                    first_attn_group = [f"add_{att_add_ids[i]}" for i in range(8)]
                    first_attn_group_bufs = [
                        f"buffer_0_{daisy_cache[4]}_{attn_op}"
                        for attn_op in first_attn_group
                    ]
                    print(
                        f"inserting daisy_chain nops between {daisy_cache[4]} and {first_attn_group_bufs}"
                    )
                    pybuda.config.insert_buffering_nop(
                        daisy_cache[4], first_attn_group_bufs, daisy_chain=True
                    )
                    [
                        pybuda.config.override_op_size(
                            f"buffer_0_{daisy_cache[4]}_buffer_0_{daisy_cache[4]}_{attn_op}",
                            (1, 1),
                        )
                        for attn_op in first_attn_group
                    ]
                    first_attn_group_bufs = [
                        f"buffer_0_{daisy_cache[4]}_{attn_op}"
                        for attn_op in first_attn_group_bufs
                    ]

                    first_layer_buffer_schd_rolled = [
                        [
                            group[i]
                            for group in [
                                first_sin_group_bufs,
                                first_cos_group_bufs,
                                first_attn_group_bufs,
                                first_read_group_k_bufs,
                                first_read_group_v_bufs,
                                first_write_group_k_bufs,
                                first_write_group_v_bufs,
                            ]
                        ]
                        for i in range(8)
                    ]
                    # put embedding for mask at chip 0
                    first_layer_buffer_schd = [
                        f"embedding_{subtract_id-3}",
                        f"input_0_subtract_{subtract_id}_splt_brcst_1_0",
                        f"input_0_subtract_{subtract_id+1}_splt_brcst_1_0",
                        f"subtract_{subtract_id}",
                    ]  #'input_0_subtract_38_splt_brcst_1_0', 'input_0_subtract_39_splt_brcst_1_0', 'subtract_38']
                    pybuda.config.override_op_placement(
                        f"embedding_{subtract_id-3}",
                        chip_id=chip_ids[self.fracture_mlp + 2],
                    )
                    pybuda.config.override_op_placement(
                        f"input_0_subtract_{subtract_id}_splt_brcst_1_0",
                        chip_id=chip_ids[self.fracture_mlp + 3],
                    )
                    [
                        first_layer_buffer_schd.extend(buf)
                        for buf in first_layer_buffer_schd_rolled
                    ]

                    first_layer_buffer_schd.append("layernorm_0.dc.reduce_sum.0.lc1")
                    pybuda.config.add_schedule_constraint(first_layer_buffer_schd)
                    [
                        pybuda.config.override_op_placement(
                            op, chip_id=chip_ids[self.fracture_mlp + 4 + 2 * mqa_id + 1]
                        )
                        for mqa_id, op in enumerate(first_sin_group_bufs)
                    ]

                    # We decrease these DFs for eltwise ops so we can reclaim DST tiles - no need for fp32 accumulate
                    pybuda.config.configure_mixed_precision(
                        op_type="multiply",
                        intermediate_df=pybuda.DataFormat.Float16_b,
                        accumulate_df=pybuda.DataFormat.Float16_b,
                    )
                    pybuda.config.configure_mixed_precision(
                        op_type="add",
                        intermediate_df=pybuda.DataFormat.Float16_b,
                        accumulate_df=pybuda.DataFormat.Float16_b,
                    )
                    pybuda.config.configure_mixed_precision(
                        op_type="subtract",
                        intermediate_df=pybuda.DataFormat.Float16_b,
                        accumulate_df=pybuda.DataFormat.Float16_b,
                    )
                    pybuda.config.configure_mixed_precision(
                        op_type="exp",
                        intermediate_df=pybuda.DataFormat.Float16_b,
                        accumulate_df=pybuda.DataFormat.Float16_b,
                    )
                    pybuda.config.configure_mixed_precision(
                        op_type="splice",
                        intermediate_df=pybuda.DataFormat.Float16_b,
                        accumulate_df=pybuda.DataFormat.Float16_b,
                    )

                    pybuda.config.configure_mixed_precision(
                        name_regex="buffer_0_subtract.*",
                        output_df=pybuda.DataFormat.Bfp2_b,
                    )

                    name_base_kv = [1, 3, 5, 7, 9, 11, 13, 15]
                    kv_update_sched_ = [[], [], [], [], [], [], [], []]
                    # mqa
                    for layer in range(len(pt_module.layers)):
                        offset = layer * offset_base
                        offset_2 = layer * 48
                        # override for layer start input broadcast
                        # broadcasting ln_mlp and ln_attn

                        # KV cache names for this layer
                        k_names = [f"k_{name_base_kv[j]+layer*48+16}" for j in range(8)]
                        v_names = [f"v_{name_base_kv[j]+layer*32}" for j in range(8)]

                        # override for mqas.
                        mqa_qmm = [
                            13,
                            88,
                            164,
                            239,
                            316,
                            391,
                            467,
                            542,
                        ]  # [13, 85, 157, 229, 301, 373, 445, 517]
                        mqa_kmm = [
                            32,
                            107,
                            183,
                            258,
                            335,
                            410,
                            486,
                            561,
                        ]  # [32, 104, 176, 248, 320, 392, 464, 536]
                        mqa_vmm = [
                            67,
                            142,
                            218,
                            293,
                            370,
                            445,
                            521,
                            596,
                        ]  # [67, 139, 211, 283, 355, 427, 499, 571]
                        # mqa_kcache = [k+end_offset for k in [31, 35, 39, 43, 47, 51, 55, 59]]
                        # mqa_vcache = [v+end_offset for v in [55, 59, 63, 67, 71, 75, 79, 83]]
                        mqa_kcache = [
                            31 + end_offset,
                            34 + end_offset,
                            37 + end_offset,
                            40 + end_offset,
                            43 + end_offset,
                            46 + end_offset,
                            49 + end_offset,
                            52 + end_offset,
                        ]
                        mqa_attmm = [
                            26,
                            101,
                            177,
                            252,
                            329,
                            404,
                            480,
                            555,
                        ]  # [26, 98, 170, 242, 314, 386, 458, 530] #[26, 97, 168, 239, 310, 381, 452, 523]
                        mqa_vcache = [
                            55 + end_offset,
                            58 + end_offset,
                            61 + end_offset,
                            64 + end_offset,
                            67 + end_offset,
                            70 + end_offset,
                            73 + end_offset,
                            76 + end_offset,
                        ]
                        mqa_end = [
                            58,
                            133,
                            209,
                            284,
                            361,
                            436,
                            512,
                            587,
                        ]  # [58, 130, 202, 274, 346, 418, 490, 562] #[58, 129, 200, 271, 342, 413, 484, 555]
                        mqa_sched = []

                        # Increasing u_kt might reduce overhead in communications
                        pybuda.config.override_u_kt(
                            f"layernorm_{0+offset}.dc.reduce_sum.0.lc1", 32
                        )
                        pybuda.config.override_u_kt(
                            f"layernorm_{10+offset}.dc.reduce_sum.0.lc1", 32
                        )
                        pybuda.config.override_u_kt(
                            f"layernorm_{0+offset}.dc.reduce_sum.5.lc1", 16
                        )
                        pybuda.config.override_u_kt(
                            f"layernorm_{10+offset}.dc.reduce_sum.5.lc1", 16
                        )

                        # per layer daisy chaining:
                        # 1. the ln output and mqa start
                        groups = [
                            [
                                f"matmul_{q+offset}",
                                f"matmul_{k+offset}",
                                f"matmul_{v+offset}",
                            ]
                            for i, (q, k, v) in enumerate(
                                zip(mqa_qmm, mqa_kmm, mqa_vmm)
                            )
                        ]
                        input = f"layernorm_{10+offset}.dc.add.14"
                        mqa_starts = insert_by_groups(
                            input, groups, verbose=False, return_intermediates=True
                        )

                        dense_sched = []
                        dense_matmul_sched = []
                        dense_nop_sched = []
                        # per mqa overrides
                        for mqa_id, (
                            start,
                            q,
                            k,
                            v,
                            kcache,
                            att,
                            vcache,
                            e,
                        ) in enumerate(
                            zip(
                                mqa_starts,
                                mqa_qmm,
                                mqa_kmm,
                                mqa_vmm,
                                mqa_kcache,
                                mqa_attmm,
                                mqa_vcache,
                                mqa_end,
                            )
                        ):
                            # place cache queue on right chip
                            pybuda.config.override_dram_queue_placement(
                                k_names[mqa_id],
                                chip_id=chip_ids[
                                    self.fracture_mlp + 4 + 2 * mqa_id + 1
                                ],
                            )
                            pybuda.config.override_dram_queue_placement(
                                v_names[mqa_id],
                                chip_id=chip_ids[
                                    self.fracture_mlp + 4 + 2 * mqa_id + 1
                                ],
                            )

                            # Insert K and V cache buffering nops
                            # k_buf = self.insert_nop(k_names[mqa_id], [f'multiply_{kcache+1+offset_2}', f'matmul_{att+offset}'])
                            k_buf = self.insert_nop(
                                k_names[mqa_id], f"matmul_{att+offset}"
                            )
                            pybuda.override_op_size(k_buf, (4, 1))
                            v_buf = self.insert_nop(
                                v_names[mqa_id], f"matmul_{e+offset}"
                            )
                            pybuda.config.override_op_size(v_buf, (1, 2))
                            v_buf = self.insert_nop(v_names[mqa_id], v_buf)
                            pybuda.config.override_op_size(v_buf, (1, 2))
                            v_buf = self.insert_nop(v_names[mqa_id], v_buf)
                            pybuda.config.override_op_size(v_buf, (1, 2))

                            # schedule each mqa in order: qkv and rotary embed
                            mqa_sched.append(start)

                            # Q matmul weight needs NOP
                            pybuda.config.override_op_size(f"matmul_{q+offset}", (1, 8))
                            q_nop = self.insert_nop(
                                f"layers.{layer}.self_attention.wq_list.{mqa_id}.weight",
                                f"matmul_{q+offset}",
                            )
                            pybuda.config.override_op_size(q_nop, (2, 8))
                            pybuda.config.internal_override_output_buffer_multiplier(
                                q_nop, multiplier=1
                            )

                            mqa_sched.append(f"matmul_{q+offset}")
                            mqa_sched.append(f"matmul_{k+offset}")
                            mqa_sched.append(f"matmul_{v+offset}")
                            mqa_sched.append(f"buffer_0_cos_multiply_{q+2+offset}")
                            mqa_sched.append(f"buffer_0_sin_multiply_{q+7+offset}")
                            mqa_sched.append(
                                f"transpose_{q+9+offset}.dc.sparse_matmul.4.lc2"
                            )
                            mqa_sched.append(
                                f"reshape_{v+1+offset}.dc.sparse_matmul.5.lc2"
                            )
                            mqa_sched.append(
                                f"transpose_{k+9+offset}.dc.sparse_matmul.3.lc2"
                            )
                            mqa_sched.extend(
                                [
                                    f"transpose_{k+9+offset}.dc.narrow.6_s_brcst_m2_0_0.lc1",
                                    f"reshape_{v+1+offset}.dc.narrow.7_s_brcst_m2_0_0.lc1",
                                ]
                            )

                            # attn matmul for new
                            mqa_sched.append(
                                f"matmul_{k+12+offset}"
                            )  # QKt matmul for new
                            mqa_sched.append(
                                f"matmul_{k+40+offset}"
                            )  # v matmul for new

                            # schedule the cache update
                            kv_update_sched_[mqa_id].extend(
                                [
                                    f"buffer_0_subtract_{subtract_id+1}_multiply_{kcache+offset_2}",
                                    f"buffer_0_subtract_{subtract_id}_multiply_{kcache+1+offset_2}",
                                    f"buffer_0_subtract_{subtract_id+1}_multiply_{vcache+offset_2}",
                                    f"buffer_0_subtract_{subtract_id}_multiply_{vcache+1+offset_2}",
                                ]
                            )
                            kv_update_sched_[mqa_id].extend(
                                [
                                    f"multiply_{kcache+offset_2}",
                                    f"multiply_{kcache+1+offset_2}",
                                    f"add_{kcache+2+offset_2}",
                                ]
                            )
                            kv_update_sched_[mqa_id].extend(
                                [
                                    f"multiply_{vcache+offset_2}",
                                    f"multiply_{vcache+1+offset_2}",
                                    f"add_{vcache+2+offset_2}",
                                ]
                            )
                            if layer % 4 == 0:
                                pybuda.config.override_op_placement(
                                    f"buffer_0_subtract_{subtract_id+1}_multiply_{kcache+offset_2}",
                                    chip_id=chip_ids[
                                        self.fracture_mlp + 4 + 2 * mqa_id + 1
                                    ],
                                )
                            pybuda.config.internal_override_output_buffer_multiplier(
                                f"buffer_0_subtract_{subtract_id+1}_multiply_{kcache+offset_2}",
                                multiplier=16,
                            )
                            pybuda.config.internal_override_output_buffer_multiplier(
                                f"buffer_0_subtract_{subtract_id}_multiply_{kcache+1+offset_2}",
                                multiplier=16,
                            )
                            pybuda.config.internal_override_output_buffer_multiplier(
                                f"buffer_0_subtract_{subtract_id+1}_multiply_{vcache+offset_2}",
                                multiplier=16,
                            )
                            pybuda.config.internal_override_output_buffer_multiplier(
                                f"buffer_0_subtract_{subtract_id}_multiply_{vcache+1+offset_2}",
                                multiplier=16,
                            )

                            # second chip, attn matmul for cache
                            mqa_sched.append(k_buf)  # buf for kT
                            mqa_sched.append(
                                f"matmul_{att+offset}"
                            )  # Qkt matmul for cache
                            mqa_sched.append(
                                f"buffer_0_mask_add_{att+3+offset}"
                            )  # attn mask buffer for cache
                            mqa_sched.append(f"matmul_{e+offset}")  # v matmul for cache
                            mqa_sched.append(
                                f"multiply_{e+24+offset}"
                            )  # last multiply op for final attn result
                            mqa_sched.append(
                                f"reshape_{e+25+offset}.dc.sparse_matmul.10.lc2"
                            )

                            #### End of schedules

                            # make write mask 1x2 and read mask 1x2
                            pybuda.override_op_size(
                                f"buffer_0_subtract_{subtract_id+1}_multiply_{kcache+offset_2}",
                                (1, 2),
                            )
                            pybuda.override_op_size(
                                f"buffer_0_subtract_{subtract_id}_multiply_{kcache+1+offset_2}",
                                (1, 2),
                            )
                            pybuda.override_op_size(
                                f"buffer_0_subtract_{subtract_id+1}_multiply_{vcache+offset_2}",
                                (1, 2),
                            )
                            pybuda.override_op_size(
                                f"buffer_0_subtract_{subtract_id}_multiply_{vcache+1+offset_2}",
                                (1, 2),
                            )

                            # make kv cache ops smaller
                            pybuda.override_op_size(
                                f"multiply_{kcache+offset_2}", (1, 2)
                            )
                            pybuda.override_op_size(
                                f"multiply_{kcache+1+offset_2}", (1, 2)
                            )
                            pybuda.override_op_size(f"add_{kcache+2+offset_2}", (1, 2))

                            pybuda.override_op_size(
                                f"multiply_{vcache+offset_2}", (1, 2)
                            )
                            pybuda.override_op_size(
                                f"multiply_{vcache+1+offset_2}", (1, 2)
                            )
                            pybuda.override_op_size(f"add_{vcache+2+offset_2}", (1, 2))

                            # QKt matmul and exp size smaller
                            pybuda.override_op_size(f"matmul_{att+offset}", (1, 8))
                            # Override attn scale multiply to 1x2 and attn mask add to 1x2
                            pybuda.override_op_size(f"multiply_{att+2+offset}", (1, 4))
                            pybuda.override_op_size(f"add_{att+3+offset}", (1, 4))
                            # softmax sizes
                            pybuda.override_op_size(f"exp_{att+27+offset}", (1, 8))
                            pybuda.override_op_size(f"subtract_{att+26+offset}", (1, 4))

                            # K, V matmul larger to decrease m_k
                            pybuda.override_op_size(f"matmul_{k+offset}", (1, 2))
                            pybuda.config.override_u_kt(f"matmul_{k+offset}", 32)
                            pybuda.override_op_size(f"matmul_{v+offset}", (1, 2))
                            pybuda.config.override_u_kt(f"matmul_{v+offset}", 32)
                            # pybuda.override_op_size(f'matmul_{q+offset}', (1,4))

                            # Override input buf sizes to increase read BW
                            pybuda.override_op_size(start, (1, 4))

                            pybuda.config.configure_mixed_precision(
                                name_regex=f"matmul_(?:{q+offset}|{v+offset}|{k+offset})",
                                input_df={
                                    1: [pybuda.DataFormat.Bfp8_b, True],
                                },
                            )

                            dense_weight_nop = self.insert_nop(
                                f"layers.{layer}.self_attention.dense_list.{mqa_id}.weight",
                                f"matmul_{e+27+offset}",
                            )
                            pybuda.config.override_op_size(dense_weight_nop, (2, 8))
                            pybuda.config.internal_override_output_buffer_multiplier(
                                dense_weight_nop, multiplier=1
                            )
                            dense_sched.append(dense_weight_nop)
                            dense_nop_sched.append(dense_weight_nop)
                            dense_matmul_sched.append(f"matmul_{e+27+offset}")
                            dense_sched.append(f"matmul_{e+27+offset}")
                            if mqa_id % 2 == 1:
                                dense_sched.append(f"add_{e+27+2+offset}")
                                pybuda.config.override_op_size(
                                    f"add_{e+27+2+offset}", (1, 8)
                                )
                            if mqa_id % 4 == 3:
                                dense_sched.append(f"add_{e+27+3+offset}")
                                pybuda.config.override_op_size(
                                    f"add_{e+27+3+offset}", (1, 8)
                                )
                            if mqa_id % 8 == 7:
                                dense_sched.append(f"add_{e+27+4+offset}")
                                pybuda.config.override_op_size(
                                    f"add_{e+27+4+offset}", (1, 8)
                                )

                            """
                            # Fork Join: values to value matmul is the short path. Want to add buffer
                            nop_name = self.insert_nop(f'add_{e+offset-4}', f'matmul_{e+offset}')
                            pybuda.config.override_op_size(nop_name, (1,2))
                            """

                            # breaking for each mqa start
                            pybuda.config.override_op_placement(
                                start,
                                chip_id=chip_ids[self.fracture_mlp + 4 + 2 * mqa_id],
                                spatial_epoch_break=True,
                            )  # HACK: removing input daisy chain
                            # pybuda.config.override_op_placement(f'matmul_{att+offset}', chip_id=chip_ids[self.fracture_mlp+4+2*mqa_id+1], spatial_epoch_break=True)
                            pybuda.config.override_op_placement(
                                k_buf,
                                chip_id=chip_ids[
                                    self.fracture_mlp + 4 + 2 * mqa_id + 1
                                ],
                                spatial_epoch_break=True,
                            )

                        mqa_sched = mqa_sched + dense_sched

                        pybuda.config.add_schedule_constraint(
                            mqa_sched + [f"concatenate_{580+offset}.dc.concatenate.8"]
                        )

                        # Put layer output on same chip as next layer's ln_attn
                        if layer != len(pt_module.layers) - 1:
                            # pybuda.config.override_dram_queue_placement(f'e2e_add_{(layer+1) * offset_base - 1}_0', chip_id=chip_ids[1])
                            pass

                        # add back later
                        # base 2 layers: 1910
                        # base 3 layers: 2831
                        # base 4 layers: 3752
                        # base 5 layers:
                        # what's the pattern?
                        # weight_id_base = 1910+ (len(pt_module.layers)-2) * 921
                        # weight_names = [f'layers.{layer}.self_attention.dense.weight', f'layers.{layer}.self_attention.dense.weight_fork_clone{weight_id_base+layer*9}',
                        #                 f'layers.{layer}.self_attention.dense.weight_fork_clone{weight_id_base+1+layer*9}',
                        #                 f'layers.{layer}.self_attention.dense.weight_fork_clone{weight_id_base+2+layer*9}']
                        # dense_weight_bufs = [self.insert_nop(weight_name, f'fractured_{i}_matmul_{591+offset}') for i, weight_name in enumerate(weight_names)]
                        # [pybuda.config.override_op_size(buf, (4,8)) for buf in dense_weight_bufs]
                        # [pybuda.config.internal_override_output_buffer_multiplier(buf, multiplier=1) for buf in dense_weight_bufs]

                        [
                            pybuda.config.override_op_placement(
                                dense_op,
                                chip_id=chip_ids[self.fracture_mlp + 4 + 2 * 8 + i],
                                spatial_epoch_break=True,
                            )
                            for i, dense_op in enumerate(dense_nop_sched[::2])
                        ]

                        # Upsize residual ops
                        pybuda.config.override_op_placement(
                            f"add_{619+offset}",
                            chip_id=chip_ids[-1],
                            spatial_epoch_break=True,
                        )
                        pybuda.config.override_op_size(f"add_{619+offset}", (1, 8))
                        pybuda.config.override_op_size(f"add_{621+offset}", (1, 8))

                        for dense_matmul in dense_matmul_sched:
                            pybuda.config.configure_mixed_precision(
                                name_regex=dense_matmul,
                                input_df={
                                    1: [pybuda.DataFormat.Bfp8_b, True],
                                },
                            )

                        if (layer + 1) % 4 == 0 or layer == len(pt_module.layers) - 1:
                            # flatten kv_update_sched_
                            kv_update_sched_ = [
                                item for sublist in kv_update_sched_ for item in sublist
                            ]
                            kv_update_sched.extend(kv_update_sched_)
                            print(
                                f"[kv cahce update schedule] extending: {kv_update_sched_}"
                            )
                            kv_update_sched_ = [[], [], [], [], [], [], [], []]

                elif (
                    num_chips == 32
                    and user_rows == 32
                    and self.fracture_mlp == 8
                    and not self.flash_decode
                ):
                    print(
                        f"32 chips {len(pt_module.layers)} layer 32 user rows {self.fracture_mlp} mlp fractures -- mlp fractured"
                    )
                    # os.environ['TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE'] = "49152"
                    # chip_ids = [0,29,27,28,31,5,4,30,3,1,25,24,23,22,21,13,14,20,19,15,16,17,18,2,26,6,7,8,9,10,11,12]
                    if self.arch == "nebula-galaxy":
                        print("Using galaxy nebula chip orders:")
                        # chip_ids = [0, 3,17,16,11,12,10,13,18,15, 1, 4,27,28,29,30,31,32,25,26, 6,24,23,22,21,20,19,7, 8, 9,14,5, 2] # 201 old
                        chip_ids = [
                            0,
                            20,
                            23,
                            22,
                            13,
                            14,
                            12,
                            15,
                            24,
                            21,
                            26,
                            19,
                            2,
                            1,
                            31,
                            32,
                            29,
                            30,
                            4,
                            3,
                            18,
                            5,
                            6,
                            28,
                            27,
                            7,
                            8,
                            9,
                            10,
                            11,
                            16,
                            17,
                            25,
                        ]  # aus-glx-1, aus-glx-02, aus-glx-03, 201_new
                        # chip_ids = [0, 21, 1, 4, 6, 5, 7, 20, 2, 3, 31, 22, 30, 29, 28, 27, 26, 25, 24, 23, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 19, 18, 32] # aus-glx-01 (old)
                        print(chip_ids)
                    else:
                        chip_ids = [
                            0,
                            3,
                            31,
                            28,
                            26,
                            27,
                            6,
                            5,
                            30,
                            29,
                            1,
                            2,
                            25,
                            24,
                            23,
                            22,
                            21,
                            20,
                            19,
                            18,
                            16,
                            15,
                            14,
                            13,
                            12,
                            11,
                            10,
                            9,
                            8,
                            7,
                            4,
                            17,
                        ]  # 206
                    # we start placing other (not mlp) ops from mlp_factor+2 chip

                    ######## manual overrides
                    offset_base = 388
                    first_layer_offset = 393
                    output_nop_schedule = []
                    daisy_cache = [
                        "sin",
                        "cos",
                        "subtract_39",
                        "subtract_38",
                        "attn_mask",
                    ]  # want daisy chain to wrap around for each layer

                    # make ensure masks saved into mmio chip 0 (if not default placed there)
                    pybuda.config.override_dram_queue_placement("attn_mask", chip_id=0)

                    def insert_by_groups(
                        input,
                        groups,
                        verbose=False,
                        return_intermediates=False,
                        override_size=None,
                    ):
                        """
                        takes in [g1, g2, g3...] and g1 = [e1, e2, e3...]
                        input -- g1 -- g2 -- g3
                        """
                        intermediates = []
                        for i in range(len(groups)):
                            sub_group = [
                                item for sublist in groups[i:] for item in sublist
                            ]  # flatten groups
                            if verbose:
                                print(
                                    "[self.insert_nops_by_groups]:",
                                    f"input: {input}",
                                    f"sub_group: {sub_group}",
                                )
                            input = self.insert_nop(input, sub_group)
                            if override_size is not None:
                                pybuda.override_op_size(input, override_size)
                            intermediates.append(input)
                        if return_intermediates:
                            return intermediates
                        else:
                            return input

                    # cross layer daisy chains
                    sin_ids = [
                        20,
                        32,
                        71,
                        83,
                        117,
                        129,
                        163,
                        175,
                        209,
                        221,
                        255,
                        267,
                        301,
                        313,
                        347,
                        359,
                    ]
                    cos_ids = [
                        15,
                        27,
                        66,
                        78,
                        112,
                        124,
                        158,
                        170,
                        204,
                        216,
                        250,
                        262,
                        296,
                        308,
                        342,
                        354,
                    ]
                    k_ids = [40, 86, 132, 178, 224, 270, 316, 362]
                    v_ids = [55, 101, 147, 193, 239, 285, 331, 377]
                    att_add_ids = [48, 94, 140, 186, 232, 278, 324, 370]
                    for i in range(8):
                        sg, cg, rg_k, rg_v, wg_k, wg_v, aag = [], [], [], [], [], [], []
                        for layer in range(len(pt_module.layers)):
                            dummy_factor = 0  # -1 if layer>0 and i>0 else 0
                            dummy_factor2 = -5 if layer > 0 and i > 0 else 0
                            dummy_factor3 = 0  # if layer>0 else 0
                            dummy_factor4 = -5 if layer > 0 else 0
                            rw_ids = [
                                [l, l + 8]
                                for l in range(layer * 16, (layer + 1) * 16 - 8)
                            ]
                            offset = (
                                0
                                if layer == 0
                                else first_layer_offset + (layer - 1) * offset_base
                            )
                            sg.extend(
                                [
                                    f"multiply_{sin_ids[2*i]+offset+dummy_factor2+dummy_factor}",
                                    f"multiply_{sin_ids[2*i+1]+offset+dummy_factor2+dummy_factor}",
                                ]
                            )
                            cg.extend(
                                [
                                    f"multiply_{cos_ids[2*i]+offset+dummy_factor2+dummy_factor}",
                                    f"multiply_{cos_ids[2*i+1]+offset+dummy_factor2+dummy_factor}",
                                ]
                            )
                            rg_k.extend(
                                [
                                    f"multiply_{k_ids[i]+1+offset+dummy_factor4+dummy_factor3}"
                                ]
                            )
                            wg_k.extend(
                                [
                                    f"multiply_{k_ids[i]+offset+dummy_factor4+dummy_factor3}"
                                ]
                            )
                            rg_v.extend(
                                [
                                    f"multiply_{v_ids[i]+1+offset+dummy_factor4+dummy_factor3}"
                                ]
                            )
                            wg_v.extend(
                                [
                                    f"multiply_{v_ids[i]+offset+dummy_factor4+dummy_factor3}"
                                ]
                            )

                            aag.extend(
                                [
                                    f"add_{att_add_ids[i]+offset+dummy_factor4+dummy_factor3}"
                                ]
                            )

                        # Insert daisy chain for each MQA going down the layers
                        pybuda.config.insert_buffering_nop(
                            daisy_cache[0], sg, hoist_tms=False, daisy_chain=True
                        )
                        print("Inserting Nops between", daisy_cache[0], sg)

                        pybuda.config.insert_buffering_nop(
                            daisy_cache[1], cg, hoist_tms=False, daisy_chain=True
                        )
                        print("Inserting Nops between", daisy_cache[1], cg)

                        pybuda.config.insert_buffering_nop(
                            daisy_cache[3], rg_k, daisy_chain=True
                        )
                        print("Inserting Nops between", daisy_cache[3], rg_k)

                        pybuda.config.insert_buffering_nop(
                            daisy_cache[3], rg_v, daisy_chain=True
                        )
                        print("Inserting Nops between", daisy_cache[3], rg_v)

                        pybuda.config.insert_buffering_nop(
                            daisy_cache[2], wg_k, daisy_chain=True
                        )
                        print("Inserting Nops between", daisy_cache[2], wg_k)

                        pybuda.config.insert_buffering_nop(
                            daisy_cache[2], wg_v, daisy_chain=True
                        )
                        print("Inserting Nops between", daisy_cache[2], wg_v)

                        pybuda.config.insert_buffering_nop(
                            daisy_cache[4], aag, daisy_chain=True
                        )
                        print("Inserting Nops between", daisy_cache[4], aag)

                        [
                            pybuda.config.override_op_size(
                                f"buffer_0_{daisy_cache[4]}_{attn_op}", (1, 4)
                            )
                            for attn_op in aag
                        ]

                    # Inserting 2D daisy chain for cos & sin leads to a strange error in `Lowering to Buda`. The error is that eltwise input must be broadcastable.
                    first_sin_group = [f"multiply_{sin_ids[2*i]}" for i in range(8)]
                    first_sin_group_bufs = [
                        f"buffer_0_{daisy_cache[0]}_{sin_op}"
                        for sin_op in first_sin_group
                    ]
                    print(
                        f"inserting daisy_chain nops between {daisy_cache[0]} and {first_sin_group_bufs}"
                    )
                    pybuda.config.insert_buffering_nop(
                        daisy_cache[0],
                        first_sin_group_bufs,
                        hoist_tms=False,
                        daisy_chain=True,
                    )
                    first_sin_group_bufs = [
                        f"buffer_0_{daisy_cache[0]}_{sin_op}"
                        for sin_op in first_sin_group_bufs
                    ]

                    first_cos_group = [f"multiply_{cos_ids[2*i]}" for i in range(8)]
                    first_cos_group_bufs = [
                        f"buffer_0_{daisy_cache[1]}_{cos_op}"
                        for cos_op in first_cos_group
                    ]
                    print(
                        f"inserting daisy_chain nops between {daisy_cache[1]} and {first_cos_group_bufs}"
                    )
                    pybuda.config.insert_buffering_nop(
                        daisy_cache[1],
                        first_cos_group_bufs,
                        hoist_tms=False,
                        daisy_chain=True,
                    )
                    first_cos_group_bufs = [
                        f"buffer_0_{daisy_cache[1]}_{cos_op}"
                        for cos_op in first_cos_group_bufs
                    ]

                    first_read_group_k = [f"multiply_{k_ids[i]+1}" for i in range(8)]
                    first_read_group_k_bufs = [
                        f"buffer_0_{daisy_cache[3]}_{read_op}"
                        for read_op in first_read_group_k
                    ]
                    print(
                        f"inserting daisy_chain nops between {daisy_cache[3]} and {first_read_group_k_bufs}"
                    )
                    pybuda.config.insert_buffering_nop(
                        daisy_cache[3], first_read_group_k_bufs, daisy_chain=True
                    )
                    [
                        pybuda.config.override_op_size(
                            f"buffer_0_{daisy_cache[3]}_buffer_0_{daisy_cache[3]}_{read_op}",
                            (2, 1),
                        )
                        for read_op in first_read_group_k
                    ]
                    first_read_group_k_bufs = [
                        f"buffer_0_{daisy_cache[3]}_{read_op}"
                        for read_op in first_read_group_k_bufs
                    ]

                    first_read_group_v = [f"multiply_{v_ids[i]+1}" for i in range(8)]
                    first_read_group_v_bufs = [
                        f"buffer_0_{daisy_cache[3]}_{read_op}"
                        for read_op in first_read_group_v
                    ]
                    print(
                        f"inserting daisy_chain nops between {daisy_cache[3]} and {first_read_group_v_bufs}"
                    )
                    pybuda.config.insert_buffering_nop(
                        daisy_cache[3], first_read_group_v_bufs, daisy_chain=True
                    )
                    [
                        pybuda.config.override_op_size(
                            f"buffer_0_{daisy_cache[3]}_buffer_0_{daisy_cache[3]}_{read_op}",
                            (1, 2),
                        )
                        for read_op in first_read_group_v
                    ]
                    first_read_group_v_bufs = [
                        f"buffer_0_{daisy_cache[3]}_{read_op}"
                        for read_op in first_read_group_v_bufs
                    ]

                    first_write_group_k = [f"multiply_{k_ids[i]}" for i in range(8)]
                    first_write_group_k_bufs = [
                        f"buffer_0_{daisy_cache[2]}_{write_op}"
                        for write_op in first_write_group_k
                    ]
                    print(
                        f"inserting daisy_chain nops between {daisy_cache[2]} and {first_write_group_k_bufs}"
                    )
                    pybuda.config.insert_buffering_nop(
                        daisy_cache[2], first_write_group_k_bufs, daisy_chain=True
                    )
                    [
                        pybuda.config.override_op_size(
                            f"buffer_0_{daisy_cache[2]}_buffer_0_{daisy_cache[2]}_{write_op}",
                            (2, 1),
                        )
                        for write_op in first_write_group_k
                    ]
                    first_write_group_k_bufs = [
                        f"buffer_0_{daisy_cache[2]}_{write_op}"
                        for write_op in first_write_group_k_bufs
                    ]

                    first_write_group_v = [f"multiply_{v_ids[i]}" for i in range(8)]
                    first_write_group_v_bufs = [
                        f"buffer_0_{daisy_cache[2]}_{write_op}"
                        for write_op in first_write_group_v
                    ]
                    print(
                        f"inserting daisy_chain nops between {daisy_cache[2]} and {first_write_group_v_bufs}"
                    )
                    pybuda.config.insert_buffering_nop(
                        daisy_cache[2], first_write_group_v_bufs, daisy_chain=True
                    )
                    [
                        pybuda.config.override_op_size(
                            f"buffer_0_{daisy_cache[2]}_buffer_0_{daisy_cache[2]}_{write_op}",
                            (1, 2),
                        )
                        for write_op in first_write_group_v
                    ]
                    first_write_group_v_bufs = [
                        f"buffer_0_{daisy_cache[2]}_{write_op}"
                        for write_op in first_write_group_v_bufs
                    ]

                    first_attn_group = [f"add_{att_add_ids[i]}" for i in range(8)]
                    first_attn_group_bufs = [
                        f"buffer_0_{daisy_cache[4]}_{attn_op}"
                        for attn_op in first_attn_group
                    ]
                    print(
                        f"inserting daisy_chain nops between {daisy_cache[4]} and {first_attn_group_bufs}"
                    )
                    pybuda.config.insert_buffering_nop(
                        daisy_cache[4], first_attn_group_bufs, daisy_chain=True
                    )
                    [
                        pybuda.config.override_op_size(
                            f"buffer_0_{daisy_cache[4]}_buffer_0_{daisy_cache[4]}_{attn_op}",
                            (1, 1),
                        )
                        for attn_op in first_attn_group
                    ]
                    first_attn_group_bufs = [
                        f"buffer_0_{daisy_cache[4]}_{attn_op}"
                        for attn_op in first_attn_group_bufs
                    ]

                    first_layer_buffer_schd_rolled = [
                        [
                            group[i]
                            for group in [
                                first_sin_group_bufs,
                                first_cos_group_bufs,
                                first_attn_group_bufs,
                                first_read_group_k_bufs,
                                first_read_group_v_bufs,
                                first_write_group_k_bufs,
                                first_write_group_v_bufs,
                            ]
                        ]
                        for i in range(8)
                    ]
                    # put embedding for mask at chip 0
                    first_layer_buffer_schd = [
                        "embedding_35",
                        "input_0_subtract_38_splt_brcst_1_0",
                        "input_0_subtract_39_splt_brcst_1_0",
                        "subtract_38",
                    ]
                    pybuda.config.override_op_placement(
                        "embedding_35", chip_id=chip_ids[self.fracture_mlp + 2]
                    )
                    pybuda.config.override_op_placement(
                        "input_0_subtract_38_splt_brcst_1_0",
                        chip_id=chip_ids[self.fracture_mlp + 3],
                    )
                    [
                        first_layer_buffer_schd.extend(buf)
                        for buf in first_layer_buffer_schd_rolled
                    ]
                    # make brcst ops bigger for subtract input for mask
                    # pybuda.config.override_op_size('input_0_subtract_38_splt_brcst_1_0_splt_brcst_3_0', (1,2))
                    # pybuda.config.override_op_size('input_0_subtract_39_splt_brcst_1_0_splt_brcst_3_0', (1,2))

                    first_layer_buffer_schd.append("layernorm_0.dc.reduce_sum.0.lc1")
                    pybuda.config.add_schedule_constraint(first_layer_buffer_schd)
                    [
                        pybuda.config.override_op_placement(
                            op, chip_id=chip_ids[self.fracture_mlp + 4 + 2 * mqa_id + 1]
                        )
                        for mqa_id, op in enumerate(first_sin_group_bufs)
                    ]

                    pybuda.override_dram_queue_placement("attn_mask", chip_id=0)

                    # We decrease these DFs for eltwise ops so we can reclaim DST tiles - no need for fp32 accumulate
                    pybuda.config.configure_mixed_precision(
                        op_type="multiply",
                        intermediate_df=pybuda.DataFormat.Float16_b,
                        accumulate_df=pybuda.DataFormat.Float16_b,
                    )
                    pybuda.config.configure_mixed_precision(
                        op_type="add",
                        intermediate_df=pybuda.DataFormat.Float16_b,
                        accumulate_df=pybuda.DataFormat.Float16_b,
                    )
                    pybuda.config.configure_mixed_precision(
                        op_type="subtract",
                        intermediate_df=pybuda.DataFormat.Float16_b,
                        accumulate_df=pybuda.DataFormat.Float16_b,
                    )
                    pybuda.config.configure_mixed_precision(
                        op_type="exp",
                        intermediate_df=pybuda.DataFormat.Float16_b,
                        accumulate_df=pybuda.DataFormat.Float16_b,
                    )
                    pybuda.config.configure_mixed_precision(
                        op_type="splice",
                        intermediate_df=pybuda.DataFormat.Float16_b,
                        accumulate_df=pybuda.DataFormat.Float16_b,
                    )

                    pybuda.config.configure_mixed_precision(
                        name_regex="buffer_0_subtract.*",
                        output_df=pybuda.DataFormat.Bfp2_b,
                    )

                    # pybuda.override_op_size('attn_mask', (1,1))

                    name_base_kv = [1, 3, 5, 7, 9, 11, 13, 15]
                    # mqa
                    for layer in range(len(pt_module.layers)):
                        offset = (
                            0
                            if layer == 0
                            else first_layer_offset + (layer - 1) * offset_base
                        )
                        # override for layer start input broadcast
                        # broadcasting ln_mlp and ln_attn

                        # KV cache names for this layer
                        k_names = [f"k_{name_base_kv[j]+layer*48+16}" for j in range(8)]
                        v_names = [f"v_{name_base_kv[j]+layer*32}" for j in range(8)]

                        # override for mqas.
                        mqa_qmm = [13, 64, 110, 156, 202, 248, 294, 340]
                        mqa_kmm = [25, 76, 122, 168, 214, 260, 306, 352]
                        mqa_vmm = [53, 99, 145, 191, 237, 283, 329, 375]
                        mqa_kcache = [40, 86, 132, 178, 224, 270, 316, 362]
                        mqa_attmm = [45, 91, 137, 183, 229, 275, 321, 367]
                        mqa_vcache = [55, 101, 147, 193, 239, 285, 331, 377]
                        mqa_end = [61, 107, 153, 199, 245, 291, 337, 383]
                        mqa_sched = []

                        # Increasing u_kt might reduce overhead in communications
                        pybuda.config.override_u_kt(
                            f"layernorm_{0+offset}.dc.reduce_sum.0.lc1", 32
                        )
                        pybuda.config.override_u_kt(
                            f"layernorm_{10+offset}.dc.reduce_sum.0.lc1", 32
                        )
                        pybuda.config.override_u_kt(
                            f"layernorm_{0+offset}.dc.reduce_sum.5.lc1", 16
                        )
                        pybuda.config.override_u_kt(
                            f"layernorm_{10+offset}.dc.reduce_sum.5.lc1", 16
                        )

                        # per layer daisy chaining:
                        # 1. the ln output and mqa start
                        dummy_factor2 = -5 if layer > 0 else 0
                        groups = [
                            [
                                f"matmul_{q+offset+(dummy_factor2 if i>0 else 0)}",
                                f"matmul_{k+offset+(dummy_factor2 if i>0 else 0)}",
                                f"matmul_{v+offset+dummy_factor2}",
                            ]
                            for i, (q, k, v) in enumerate(
                                zip(mqa_qmm, mqa_kmm, mqa_vmm)
                            )
                        ]
                        input = f"layernorm_{10+offset}.dc.add.14"
                        mqa_starts = insert_by_groups(
                            input, groups, verbose=False, return_intermediates=True
                        )

                        # per mqa overrides
                        for mqa_id, (
                            start,
                            q,
                            k,
                            v,
                            kcache,
                            att,
                            vcache,
                            e,
                        ) in enumerate(
                            zip(
                                mqa_starts,
                                mqa_qmm,
                                mqa_kmm,
                                mqa_vmm,
                                mqa_kcache,
                                mqa_attmm,
                                mqa_vcache,
                                mqa_end,
                            )
                        ):
                            dummy_factor = -1 if layer == 0 and mqa_id == 0 else 0
                            dummy_factor2 = -5 if layer > 0 else 0
                            dummy_factor3 = -5 if layer > 0 and mqa_id > 0 else 0

                            # schedule each mqa in order: qkv, kcache, att, vcache, end
                            mqa_sched.append(start)
                            mqa_sched.append(f"matmul_{q+offset+dummy_factor3}")
                            mqa_sched.append(f"matmul_{k+offset+dummy_factor3}")
                            mqa_sched.append(f"matmul_{v+offset+dummy_factor2}")
                            mqa_sched.append(
                                f"buffer_0_cos_multiply_{q+2+offset+dummy_factor3}"
                            )
                            mqa_sched.append(
                                f"buffer_0_sin_multiply_{q+7+offset+dummy_factor3}"
                            )
                            mqa_sched.append(
                                f"transpose_{q+9+offset+dummy_factor3}.dc.sparse_matmul.4.lc2"
                            )
                            mqa_sched.append(
                                f"reshape_{v+1+offset+dummy_factor2}.dc.narrow.7_s_brcst_m2_0_0.lc1"
                            )
                            mqa_sched.append(
                                f"transpose_{k+9+offset+dummy_factor3}.dc.sparse_matmul.3.lc2"
                            )

                            mqa_sched.extend(
                                [
                                    f"buffer_0_subtract_39_multiply_{kcache+offset+dummy_factor2}",
                                    f"buffer_0_subtract_38_multiply_{kcache+1+offset+dummy_factor2}",
                                    f"buffer_0_subtract_39_multiply_{vcache+offset+dummy_factor2}",
                                    f"buffer_0_subtract_38_multiply_{vcache+1+offset+dummy_factor2}",
                                ]
                            )
                            # make write mask 1x2 and read mask 1x2
                            pybuda.override_op_size(
                                f"buffer_0_subtract_39_multiply_{kcache+offset+dummy_factor2}",
                                (2, 1),
                            )
                            pybuda.override_op_size(
                                f"buffer_0_subtract_38_multiply_{kcache+1+offset+dummy_factor2}",
                                (2, 1),
                            )
                            pybuda.override_op_size(
                                f"buffer_0_subtract_39_multiply_{vcache+offset+dummy_factor2}",
                                (1, 2),
                            )
                            pybuda.override_op_size(
                                f"buffer_0_subtract_38_multiply_{vcache+1+offset+dummy_factor2}",
                                (1, 2),
                            )
                            # mqa_sched.append(k_buf)
                            mqa_sched.extend(
                                [
                                    f"multiply_{kcache+offset+dummy_factor2}",
                                    f"multiply_{kcache+1+offset+dummy_factor2}",
                                    f"add_{kcache+2+offset+dummy_factor2}",
                                ]
                            )

                            mqa_sched.append(
                                f"buffer_0_attn_mask_add_{kcache+8+offset+dummy_factor2}"
                            )
                            # mqa_sched.append(v_buf)
                            mqa_sched.extend(
                                [
                                    f"multiply_{vcache+offset+dummy_factor2}",
                                    f"multiply_{vcache+1+offset+dummy_factor2}",
                                    f"add_{vcache+2+offset+dummy_factor2}",
                                ]
                            )

                            mqa_sched.append(f"matmul_{att+offset+dummy_factor2}")
                            mqa_sched.append(f"matmul_{e+offset+dummy_factor2}")

                            # make kv cache ops smaller
                            pybuda.override_op_size(
                                f"multiply_{kcache+offset+dummy_factor2}", (2, 1)
                            )
                            pybuda.override_op_size(
                                f"multiply_{kcache+1+offset+dummy_factor2}", (2, 1)
                            )
                            pybuda.override_op_size(
                                f"add_{kcache+2+offset+dummy_factor2}", (2, 2)
                            )

                            pybuda.override_op_size(
                                f"multiply_{vcache+offset+dummy_factor2}", (1, 2)
                            )
                            pybuda.override_op_size(
                                f"multiply_{vcache+1+offset+dummy_factor2}", (1, 2)
                            )
                            pybuda.override_op_size(
                                f"add_{vcache+2+offset+dummy_factor2}", (1, 2)
                            )

                            # QKt matmul
                            pybuda.override_op_size(
                                f"matmul_{att+offset+dummy_factor2}", (1, 2)
                            )
                            # Override attn scale multiply to 1x2 and attn mask add to 1x2
                            pybuda.override_op_size(
                                f"multiply_{att+2+offset+dummy_factor2}", (1, 2)
                            )
                            pybuda.override_op_size(
                                f"add_{att+3+offset+dummy_factor2}", (1, 2)
                            )
                            # softmax sizes
                            pybuda.override_op_size(
                                f"softmax_{att+4+offset+dummy_factor2}.dc.exp.2", (1, 8)
                            )
                            pybuda.override_op_size(
                                f"softmax_{att+4+offset+dummy_factor2}.dc.subtract.1",
                                (1, 4),
                            )

                            # K, V matmul larger to decrease m_k
                            pybuda.override_op_size(
                                f"matmul_{k+offset+dummy_factor3}", (1, 2)
                            )
                            pybuda.config.override_u_kt(
                                f"matmul_{k+offset+dummy_factor3}", 32
                            )
                            pybuda.override_op_size(
                                f"matmul_{v+offset+dummy_factor2}", (1, 2)
                            )
                            pybuda.config.override_u_kt(
                                f"matmul_{v+offset+dummy_factor2}", 32
                            )

                            # Override input buf sizes to increase read BW
                            pybuda.override_op_size(start, (1, 4))

                            pybuda.config.configure_mixed_precision(
                                name_regex=f"matmul_(?:{q+offset+dummy_factor3}|{v+offset+dummy_factor2}|{k+offset+dummy_factor3})",
                                input_df={
                                    1: [pybuda.DataFormat.Bfp8_b, True],
                                },
                            )

                            # Fork Join: values to value matmul is the short path. Want to add buffer
                            nop_name = self.insert_nop(
                                f"add_{e+offset-4+dummy_factor2}",
                                f"matmul_{e+offset+dummy_factor2}",
                            )
                            pybuda.config.override_op_size(nop_name, (1, 2))

                            # breaking for each mqa start
                            pybuda.config.override_op_placement(
                                start,
                                chip_id=chip_ids[self.fracture_mlp + 4 + 2 * mqa_id],
                                spatial_epoch_break=True,
                            )  # HACK: removing input daisy chain
                            pybuda.config.override_op_placement(
                                f"reshape_{v+1+offset+dummy_factor2}.dc.narrow.7_s_brcst_m2_0_0.lc1",
                                chip_id=chip_ids[
                                    self.fracture_mlp + 4 + 2 * mqa_id + 1
                                ],
                                spatial_epoch_break=True,
                            )

                        pybuda.config.add_schedule_constraint(mqa_sched)

                        # Put layer output on same chip as next layer's ln_attn
                        if layer != len(pt_module.layers) - 1:
                            # Changing this again to place on MLP chip
                            pybuda.config.override_dram_queue_placement(
                                f"e2e_add_{first_layer_offset+layer * offset_base - 1}_0",
                                chip_id=chip_ids[1],
                            )
                            pass

                        dummy_factor = (
                            5 if layer == 0 else 0
                        )  # first layer is special :(

                        # add fracturing the attn dense matmul
                        dense_factor = 4
                        # pybuda.config.insert_fracture_group([(f"matmul_{383+offset}", pybuda.k_dim, dense_factor)])
                        pybuda.config.insert_fracture_group(
                            [(f"matmul_{383+offset+dummy_factor}", -1, dense_factor)]
                        )
                        fracture_bufs = [
                            self.insert_nop(
                                f"reshape_{381+offset+dummy_factor}.dc.sparse_matmul.10.lc2",
                                f"fractured_{i}_matmul_{383+offset+dummy_factor}",
                                daisy_chain=True,
                            )
                            for i in range(dense_factor)
                        ]
                        pybuda.config.internal_override_output_buffer_multiplier(
                            f"fractured_gather_n0_matmul_{383+offset+dummy_factor}.dc.concatenate.0",
                            multiplier=1,
                        )

                        # Decrease fractured matmul output buffering and increase weight input buffering
                        [
                            pybuda.config.internal_override_output_buffer_multiplier(
                                f"fractured_{i}_matmul_{383+offset+dummy_factor}",
                                multiplier=1,
                            )
                            for i in range(dense_factor)
                        ]
                        # [pybuda.config.override_input_buffer_multiplier(f'fractured_{i}_matmul_{383+offset}', operand_index=1, multiplier=1) for i in range(dense_factor)]

                        # Upsize buffers to 1x8 for faster data transfer
                        [
                            pybuda.config.override_op_size(buf, (1, 8))
                            for buf in fracture_bufs
                        ]

                        # Alternate fracture_bufs and fractured matmuls in a list
                        fracture_sched = []
                        for i in range(dense_factor):
                            fracture_sched.append(fracture_bufs[i])
                            fracture_sched.append(
                                f"fractured_{i}_matmul_{383+offset+dummy_factor}"
                            )
                        pybuda.config.add_schedule_constraint(fracture_sched)

                        # Increase matmul weight input buffering
                        # [pybuda.config.override_input_buffer_multiplier(f'fractured_{i}_matmul_{383+offset}', operand_index=1, multiplier=1) for i in range(dense_factor)]

                        pybuda.config.override_op_placement(
                            fracture_bufs[0],
                            chip_id=chip_ids[self.fracture_mlp + 4 + 2 * 8 + 1],
                            spatial_epoch_break=True,
                        )
                        pybuda.config.override_op_placement(
                            fracture_bufs[1],
                            chip_id=chip_ids[self.fracture_mlp + 4 + 2 * 8 + 2],
                            spatial_epoch_break=True,
                        )
                        pybuda.config.override_op_placement(
                            fracture_bufs[2],
                            chip_id=chip_ids[self.fracture_mlp + 4 + 2 * 8 + 3],
                            spatial_epoch_break=True,
                        )
                        pybuda.config.override_op_placement(
                            fracture_bufs[3],
                            chip_id=chip_ids[self.fracture_mlp + 4 + 2 * 8 + 4],
                            spatial_epoch_break=True,
                        )

                        # Buffers in front of dense concat on concat chip for remote fractures (all but the last one)
                        concat_bufs = [
                            self.insert_nop(
                                f"fractured_{i}_matmul_{383+offset+dummy_factor}",
                                f"fractured_gather_n0_matmul_{383+offset+dummy_factor}.dc.concatenate.0",
                            )
                            for i in range(dense_factor - 1)
                        ]
                        [
                            pybuda.config.override_op_size(buf, (1, 4))
                            for buf in concat_bufs
                        ]

                        # add back later
                        # base 2 layers: 1286
                        # base 3 layers: 1895
                        # base 4 layers: 2504
                        # base 5 layers: 3113
                        # what's the pattern?
                        weight_id_base = 1286 + (len(pt_module.layers) - 2) * 609
                        weight_names = [
                            f"layers.{layer}.self_attention.dense.weight",
                            f"layers.{layer}.self_attention.dense.weight_fork_clone{weight_id_base+layer*9}",
                            f"layers.{layer}.self_attention.dense.weight_fork_clone{weight_id_base+1+layer*9}",
                            f"layers.{layer}.self_attention.dense.weight_fork_clone{weight_id_base+2+layer*9}",
                        ]
                        dense_weight_bufs = [
                            self.insert_nop(
                                weight_name,
                                f"fractured_{i}_matmul_{383+offset+dummy_factor}",
                            )
                            for i, weight_name in enumerate(weight_names)
                        ]
                        [
                            pybuda.config.override_op_size(buf, (4, 8))
                            for buf in dense_weight_bufs
                        ]
                        [
                            pybuda.config.internal_override_output_buffer_multiplier(
                                buf, multiplier=1
                            )
                            for buf in dense_weight_bufs
                        ]
                        concat_buf_sched = [
                            f"fractured_3_matmul_{383+offset+dummy_factor}"
                        ] + concat_bufs
                        pybuda.config.add_schedule_constraint(concat_buf_sched)

                        # Upsize residual ops
                        pybuda.config.override_op_size(
                            f"add_{385+offset+dummy_factor}", (1, 8)
                        )
                        pybuda.config.override_op_size(
                            f"add_{387+offset+dummy_factor}", (1, 8)
                        )

                        # Last reshape is bottleneck
                        pybuda.override_op_size(
                            f"reshape_{381+offset+dummy_factor}.dc.sparse_matmul.10.lc2",
                            (1, 8),
                        )

                        # Concat placement override
                        pybuda.config.override_op_placement(
                            f"concatenate_{380+offset+dummy_factor}.dc.concatenate.8",
                            chip_id=chip_ids[self.fracture_mlp + 4 + 2 * 8],
                            spatial_epoch_break=True,
                        )

                        pybuda.config.configure_mixed_precision(
                            name_regex=f".*matmul_{383+offset+dummy_factor}",
                            input_df={
                                1: [pybuda.DataFormat.Bfp8_b, True],
                            },
                        )

                if od_lm_head and not fracture_vocab:
                    print("on-device lm head override")
                    if self.flash_decode:
                        lm_head_offset = len(pt_module.layers) * 588
                    else:
                        lm_head_offset = 393 + (len(pt_module.layers) - 1) * 388
                    # start in a new temporal epoch
                    pybuda.config.override_op_placement(
                        f"layernorm_{lm_head_offset}.dc.reduce_sum.0.lc1",
                        chip_id=chip_ids[-1 if self.arch == "nebula-galaxy" else 0],
                        temporal_epoch_break=True,
                    )
                    # fracture the lm-head matmul: matmul_{lm_head_offset+3}
                    fracture_factor = 32
                    pybuda.config.insert_fracture_group(
                        [
                            # Can't do fracturing of weights due to transpose
                            (
                                f"matmul_{lm_head_offset+3}",
                                pybuda.k_dim,
                                fracture_factor,
                            ),
                        ]
                    )
                    constr = self.add_sched_interactive_mlp(
                        pybuda,
                        ops=[f"matmul_{lm_head_offset+3}"],
                        fracture_factor=fracture_factor,
                        constr=[],
                        chip_ids=chip_ids[-fracture_factor:],
                        schedule_reduced_ops=True,
                    )
                    pybuda.config.add_schedule_constraint(constr)

                elif od_lm_head and fracture_vocab:
                    print("on-device lm head override with fractured vocab")
                    if self.flash_decode:
                        lm_head_offset = len(pt_module.layers) * 622
                    else:
                        lm_head_offset = 393 + (len(pt_module.layers) - 1) * 388
                    fracture_factor = 4
                    constr = []
                    # start in a new temporal epoch
                    # pybuda.config.override_op_placement(f'layernorm_{lm_head_offset}.dc.reduce_sum.0.lc1', chip_id=chip_ids[1], temporal_epoch_break=True)
                    for i in range(fracture_vocab_factor):
                        # pybuda.config.override_op_placement(f'matmul_{lm_head_offset+3+3*i}', chip_id=(chip_ids+[0])[1+i*4], spatial_epoch_break=True)
                        pybuda.config.insert_fracture_group(
                            [
                                # Can't do fracturing of weights due to transpose
                                (
                                    f"matmul_{lm_head_offset+3+3*i}",
                                    pybuda.k_dim,
                                    fracture_factor,
                                ),
                            ]
                        )
                        constr += self.add_sched_interactive_mlp(
                            pybuda,
                            ops=[f"matmul_{lm_head_offset+3+3*i}"],
                            fracture_factor=fracture_factor,
                            constr=[],
                            chip_ids=(chip_ids + [0])[1 + i * 4 : 1 + (i + 1) * 4],
                            schedule_reduced_ops=True,
                        )

                    # place output nops onto new epoch
                    output_nop_ids = [
                        lm_head_offset + 3 + 3 * i for i in range(fracture_vocab_factor)
                    ]
                    for op_id in output_nop_ids:
                        constr.append(
                            f"fractured_gather_k0_matmul_{op_id}_cascade_sink_output_nop_0"
                        )
                    pybuda.config.override_op_placement(
                        f"fractured_gather_k0_matmul_{output_nop_ids[0]}_cascade_sink_output_nop_0",
                        chip_id=0,
                        spatial_epoch_break=True,
                    )
                    pybuda.config.add_schedule_constraint(constr + kv_update_sched)

        self.chip_ids = chip_ids

        if self.fracture_mlp > 0:
            if self.version == "efficient-40b":
                self.fracture_mlp_group_efficient40b(
                    pybuda, self.fracture_mlp, self.chip_ids
                )

    def get_chip_ids(self):
        return self.chip_ids

    def fracture_mlp_group_efficient40b(self, pybuda, fracture_mlp, chip_ids):
        mlp_constr = []
        mlp_factor = fracture_mlp
        if self.flash_decode:
            h4h_offset = 622
            first_layer_offset = 622
        else:
            h4h_offset = 388
            first_layer_offset = 393

        for layer_num in range(len(self.pt_module.layers)):
            offset = (
                0
                if layer_num == 0
                else first_layer_offset + (layer_num - 1) * h4h_offset
            )
            print(
                f"MLP Fracture: Layer: {layer_num}, matmul offset = {3+offset} & {8+offset}"
            )
            ops = [f"matmul_{3+offset}", f"gelu_{5+offset}", f"matmul_{8+offset}"]

            # MLP fracture
            pybuda.config.insert_fracture_group(
                [
                    # Can't do fracturing of weights due to transpose
                    (f"matmul_{3+offset}", -1, mlp_factor),
                    (f"matmul_{8+offset}", pybuda.k_dim, mlp_factor),
                ]
            )

            pybuda.config.configure_mixed_precision(
                name_regex=f".*matmul_(?:{3+offset}|{8+offset})",
                input_df={
                    1: [pybuda.DataFormat.Bfp8_b, True],
                },
            )

            if layer_num > 0:
                ln_mlp_nop = self.insert_nop(
                    f"add_{offset-1}",
                    [
                        f"layernorm_{0+offset}.dc.reduce_sum.0.lc1",
                        f"layernorm_{0+offset}.dc.subtract.3",
                    ],
                    daisy_chain=True,
                )
                pybuda.config.override_op_size(ln_mlp_nop, (1, 4))
                ln_mlp_nop_1 = self.insert_nop(f"add_{offset-1}", ln_mlp_nop)
                pybuda.config.override_op_size(ln_mlp_nop_1, (1, 4))
                mlp_constr += [ln_mlp_nop_1, ln_mlp_nop]

            input_name = f"layernorm_{0+offset}.dc.add.14"

            # Increase size of LN ops
            pybuda.config.override_op_size(
                f"layernorm_{0+offset}.dc.multiply.2", (1, 4)
            )
            pybuda.config.override_op_size(
                f"layernorm_{0+offset}.dc.subtract.3", (1, 4)
            )
            pybuda.config.override_op_size(
                f"layernorm_{0+offset}.dc.multiply.4", (1, 4)
            )
            pybuda.config.override_op_size(
                f"layernorm_{0+offset}.dc.multiply.12", (1, 4)
            )
            pybuda.config.override_op_size(
                f"layernorm_{0+offset}.dc.multiply.13", (1, 4)
            )
            pybuda.config.override_op_size(f"layernorm_{0+offset}.dc.add.14", (1, 4))
            pybuda.config.override_op_size(
                f"layernorm_{10+offset}.dc.multiply.2", (1, 4)
            )
            pybuda.config.override_op_size(
                f"layernorm_{10+offset}.dc.subtract.3", (1, 4)
            )
            pybuda.config.override_op_size(
                f"layernorm_{10+offset}.dc.multiply.4", (1, 4)
            )
            pybuda.config.override_op_size(
                f"layernorm_{10+offset}.dc.multiply.12", (1, 4)
            )
            pybuda.config.override_op_size(
                f"layernorm_{10+offset}.dc.multiply.13", (1, 4)
            )
            pybuda.config.override_op_size(f"layernorm_{10+offset}.dc.add.14", (1, 4))

            mlp_constr += [
                f"layernorm_{0+offset}.dc.reduce_sum.0.lc1",
                f"layernorm_{0+offset}.dc.add.14",
            ]
            mlp_constr = self.add_sched_interactive_mlp(
                pybuda,
                ops,
                mlp_factor,
                mlp_constr,
                chip_ids[2 : mlp_factor + 2],
                schedule_reduced_ops=(mlp_factor & (mlp_factor - 1) == 0),
                inp_name=input_name,
            )

            if layer_num > 0:
                ln_attn_nop = self.insert_nop(
                    f"add_{offset-1}",
                    [
                        f"layernorm_{10+offset}.dc.reduce_sum.0.lc1",
                        f"layernorm_{10+offset}.dc.subtract.3",
                    ],
                    daisy_chain=True,
                )
                pybuda.config.override_op_size(ln_attn_nop, (1, 4))
                mlp_constr += [ln_attn_nop]
            mlp_constr += [f"layernorm_{10+offset}.dc.reduce_sum.0.lc1"]

            if layer_num == 0:
                pybuda.config.override_op_placement(
                    f"layernorm_{0+offset}.dc.reduce_sum.0.lc1",
                    chip_id=chip_ids[1],
                    spatial_epoch_break=True,
                )
                pybuda.config.override_op_placement(
                    f"layernorm_{10+offset}.dc.reduce_sum.0.lc1",
                    chip_id=chip_ids[mlp_factor + 2],
                    spatial_epoch_break=True,
                )
            else:
                pybuda.config.override_op_placement(
                    ln_mlp_nop, chip_id=chip_ids[1], temporal_epoch_break=True
                )
                pybuda.config.override_op_placement(
                    ln_attn_nop,
                    chip_id=chip_ids[mlp_factor + 2],
                    spatial_epoch_break=True,
                )

        pybuda.config.add_schedule_constraint(mlp_constr)

    def insert_nop(self, input, consumers, daisy_chain=False, hoist_tms=False):
        self.pybuda.config.insert_buffering_nop(
            input, consumers, hoist_tms=hoist_tms, daisy_chain=daisy_chain
        )
        consumer = consumers[0] if isinstance(consumers, list) else consumers
        return f"buffer_0_{input}_{consumer}"

    def add_sched_interactive_mlp(
        self,
        pybuda,
        ops,
        fracture_factor,
        constr,
        chip_ids,
        schedule_reduced_ops=False,
        inp_name=None,
    ):
        depth = None
        depth_count = None
        if schedule_reduced_ops:
            # assert fracture factor is a power of 2
            assert (
                fracture_factor & (fracture_factor - 1) == 0
            ), "Fracture factor must be a power of 2"
            depth = int(np.log2(fracture_factor))
            assert depth > 0, "Invalid depth, Check the fracture factor"
            depth_count = [0 for _ in range(depth)]
        for f in range(fracture_factor):
            for op_num, op in enumerate(ops):
                fop = f"fractured_{f}_{op}"
                if "gelu" in op:
                    pybuda.config.override_op_size(
                        f"fractured_{f}_{op}", (1, 8)
                    )  # make gelu bigger
                if op_num == 0:
                    if inp_name is not None:
                        inp_nop = self.insert_nop(inp_name, fop)
                        pybuda.config.override_op_size(inp_nop, (1, 4))
                        constr.append(inp_nop)
                    print(
                        f"[add_sched_interactive]: Override op placement: {fop}, chip {chip_ids[f]}"
                    )
                    pybuda.config.override_op_placement(
                        fop if inp_name is None else inp_nop, chip_id=chip_ids[f]
                    )
                constr.append(fop)
            op = ops[-1]
            if schedule_reduced_ops:
                # schedule reduced ops
                for d in range(depth):
                    if (f + 1) % (2 ** (d + 1)) == 0:
                        rop = f"fractured_gather_k0_{op}" + d * "_cascade_sink"
                        if d != depth - 1:
                            rop = rop + f"_cascade_{depth_count[d]}"

                        # Upsize reduce ops
                        pybuda.config.override_op_size(rop, (1, 8))
                        print(
                            f"[add_sched_interactive] Override op placement: {rop}, chip {chip_ids[f]}"
                        )
                        constr.append(rop)
                        depth_count[d] += 1
        return constr
