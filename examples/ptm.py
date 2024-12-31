"""
This example shows how to use Ray Data for running offline batch inference
distributively on a multi-nodes cluster.

Learn more about Ray Data in https://docs.ray.io/en/latest/data/data.html
"""

from typing import Any, Dict, List

import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from termcolor import cprint
from vllm import LLM, SamplingParams
import torch
from vllm.ptm_utils import *

assert Version(ray.__version__) >= Version(
    "2.22.0"), "Ray version must be at least 2.22.0"

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Set tensor parallelism per instance.
tensor_parallel_size = 8

# Set number of instances. Each instance will use tensor_parallel_size GPUs.
num_instances = 1

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


# info got from ptm
from vllm import ptm_utils
torch.distributed.init_process_group()
ptm_utils.vllm_tp_gpu_group_ = torch.distributed.new_group([i for i in range(8)], backend="nccl")
ptm_utils.vllm_tp_cpu_group_ = torch.distributed.new_group([i for i in range(8)], backend="gloo")

# Create a class to do batch inference.
class LLMPredictor:

    def __init__(self):
        # Create an LLM.
        if is_driver_rank():
            self.llm = LLM(model="meta-llama/Llama-2-7b-chat-hf",
                        tensor_parallel_size=tensor_parallel_size,
                        distributed_executor_backend="mp")
        else:
            # act as vllm worker
            worker = create_worker()
            worker.start_worker_execution_loop()
            release_worker_gpu_memory(worker)
            debug_print()

    def __call__(self, batch=None) -> Dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = self.llm.generate(prompts, sampling_params)
        prompt: List[str] = []
        generated_text: List[str] = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append(' '.join([o.text for o in output.outputs]))
        return {
            "prompt": prompt,
            "generated_text": generated_text,
        }


llm = LLMPredictor()
while 1:
    if is_driver_rank():
        outputs = llm()
        cprint(outputs, "red", flush=True)
        # import time
        # time.sleep(10000)
cprint(f"rank is {torch.distributed.get_rank()}, job done\n", "red", flush=True)