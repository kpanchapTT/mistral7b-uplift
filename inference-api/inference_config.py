from collections import namedtuple
import os
InferenceConfig = namedtuple('InferenceConfig', ['hf_cache', 'number_layers',])

inference_config = InferenceConfig(
    hf_cache="/proj_sw/large-model-cache/falcon40b",
    number_layers=1
)