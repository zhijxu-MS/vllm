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
            self.worker = self.llm.llm_engine.model_executor.driver_worker
        else:
            # act as vllm worker
            self.worker = create_worker()

    def run_forward(self, prompts=None):
        if is_driver_rank():
            responses = self(prompts)
            return responses
        else:
            self.worker.start_worker_execution_loop()

    def resetup(self, weight_dict=None):
        # self.worker = create_worker()
        if weight_dict is not None:
            assert set(weight_dict.keys()) == set(worker.model_runner.model.state_dict().keys()), "something wrong, model parameters key name mismatched"
            self.worker.model_runner.model.load_state_dict(weight_dict)
        self.worker.model_runner.model.to(self.worker.model_runner.model.original_device)
        self.worker.initialize_cache(**self.worker.initialize_cache_info)

    def release_gpu_memory(self):
        for cache_engine in self.worker.cache_engine:
            cache_engine.gpu_cache.clear()
        self.worker.model_runner.model.original_device = list(self.worker.model_runner.model.parameters())[0].device
        self.worker.model_runner.model.to("cpu")
        print(f"rank is {torch.distributed.get_rank()}, mem in used {torch.cuda.memory_allocated()/1024/1024/1024}GB")

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
        res = llm.run_forward(prompts)
        if is_driver_rank():
            cprint(res, "red", flush=True)
        # llm.release_gpu_memory()
        # llm.resetup()
        # import time
        # time.sleep(10000)
cprint(f"rank is {torch.distributed.get_rank()}, job done\n", "red", flush=True)