import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PrefillConfig:
    pybuda: Any = None
    pt_module: Any = None
    version: str = None
    log_level: str = None
    fracture_mlp: int = 0

    default_df_override: Any = None
    accumulate_df: Any = None
    amp_level: int = 0
    enable_auto_fusing: bool = False
    performance_trace: Any = None
    backend_opt_level: int = 0
    enable_auto_transposing_placement: bool = True
    enable_t_streaming: bool = True
    manual_t_streaming: bool = True
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
        os.environ["PYBUDA_ENABLE_OUTPUT_QUEUES_ON_HOST"] = "1"
        os.environ["PYBUDA_DISABLE_DYNAMIC_DRAM"] = "1"
        # we want to force thread
        os.environ["PYBUDA_FORCE_THREADS"] = "1"
        os.environ["TT_BACKEND_EPOCH_BIN_NUM_SLOTS"] = "128"
        os.environ["TT_BACKEND_POP_TIMEOUT"] = "500"
        os.environ["TT_BACKEND_PUSH_TIMEOUT"] = "500"
        os.environ["TT_BACKEND_GET_TIMEOUT"] = "5000"
        os.environ["PYBUDA_DISABLE_DRAM0"] = "1"

        if self.log_level:
            os.environ["LOGGER_LEVEL"] = self.log_level
            os.environ["LOGURU_LEVEL"] = self.log_level

        if self.queues_on_host:
            os.environ["PYBUDA_ENABLE_OUTPUT_QUEUES_ON_HOST"] = "0"

        if self.arch == "nebula-galaxy":
            os.environ["TT_BACKEND_HETEROGENOUS_HARVESTING"] = "0"

    def placement_overrides(self):
        pybuda = self.pybuda
        offset = 196
        [
            self.pybuda.config.internal_override_output_buffer_multiplier(
                f"concatenate_{188+i*offset}", multiplier=1
            )
            for i in range(len(self.pt_module.layers))
        ]

        pybuda.config.configure_mixed_precision(
            name_regex="sparse_matmul",
            output_df=pybuda.DataFormat.Float16_b,
            intermediate_df=pybuda.DataFormat.Float16_b,
        )

        if self.fracture_mlp > 0:
            self.fracture_mlp_group(pybuda, self.fracture_mlp, self.chip_ids)

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

    def get_chip_ids(self):
        return self.chip_ids

    def fracture_mlp_group(self, pybuda, fracture_mlp, chip_ids):
        mlp_constr = []
        mlp_factor = fracture_mlp
        h4h_offset = 196  # FIXME

        for layer_num in range(len(self.pt_module.layers)):
            mlp_offset = layer_num * h4h_offset + (1 if layer_num > 0 else 0)
            print(
                f"MLP Fracture: Layer: {layer_num}, matmul offset = {3+mlp_offset} & {8+mlp_offset}"
            )
            ops = [
                f"matmul_{3+mlp_offset}",
                f"gelu_{5+mlp_offset}",
                f"matmul_{8+mlp_offset}",
            ]

            # MLP fracture
            pybuda.config.insert_fracture_group(
                [
                    # Can't do fracturing of weights due to transpose
                    (f"matmul_{3+mlp_offset}", -1, mlp_factor),
                    (f"matmul_{8+mlp_offset}", pybuda.k_dim, mlp_factor),
                ]
            )

            # TODO Entries/Exits is not needed anymore. Delete when we're happy with the schedule function.
            # layernorm before mlp
            mlp_constr.append(f"layernorm_{0+mlp_offset}.dc.reduce_sum.0.lc1")
            # mlp schedule
            # mlp_constr = self.add_sched_interactive_mlp(pybuda, ops, mlp_factor, mlp_constr, chip_ids[2:mlp_factor+2], schedule_reduced_ops=(mlp_factor & (mlp_factor - 1) == 0))
            # layernorm after mlp
            mlp_constr.append(f"layernorm_{10+mlp_offset}.dc.reduce_sum.0.lc1")
            # pybuda.config.override_op_placement(f'layernorm_{10+mlp_offset}.dc.reduce_sum.0.lc1', chip_id=chip_ids[mlp_factor+2], spatial_epoch_break=True)

        pybuda.config.add_schedule_constraint(mlp_constr)
