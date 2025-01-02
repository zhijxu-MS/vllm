import torch
from termcolor import cprint

import builtins

vllm_tp_cpu_group_ = None
vllm_tp_gpu_group_ = None

def is_driver_rank():
    return torch.distributed.get_rank() == 0

def get_info_from_driver_rank(tp_cpu_group=vllm_tp_cpu_group_):
    obj = [None]
    torch.distributed.broadcast_object_list(obj, src=0, group=tp_cpu_group)
    return obj[0]

def broadcast_info_to_non_driver_rank(obj, tp_cpu_group=vllm_tp_cpu_group_):
    torch.distributed.broadcast_object_list([obj], src=0, group=tp_cpu_group)

def send_info_to_non_driver_rank(obj, dst, tp_cpu_group=vllm_tp_cpu_group_):
    torch.distributed.send_object_list([obj], dst=dst, group=tp_cpu_group)

def get_info_from_driver_rank(tp_cpu_group=vllm_tp_cpu_group_):
    obj = [None]
    torch.distributed.recv_object_list(obj, src=0, group=tp_cpu_group)
    return obj[0]

def create_worker():
    from vllm.executor.gpu_executor import create_worker
    info = get_info_from_driver_rank()
    worker = create_worker(**info)
    worker.init_device()
    worker.load_model()
    worker.determine_num_available_blocks()
    info = get_info_from_driver_rank()
    kwargs = info['kwargs']
    # create kv cache, and then create cuda graph
    # TODO cuda graph also occupy gpu mem, should release it when calling release_gpu_memory
    worker.initialize_cache(**kwargs)
    return worker

def release_worker_gpu_memory(worker):
    for cache_engine in worker.cache_engine:
        cache_engine.gpu_cache.clear()
    worker.model_runner.model.original_device = worker.model_runner.model.device
    worker.model_runner.model.to("cpu")
    print(f"rank is {torch.distributed.get_rank()}, mem in used {torch.cuda.memory_allocated()/1024/1024/1024}GB")

def resetup_worker(weight_dict):
    # TODO if create_worker takes time, then refine this function
    worker = create_worker()
    assert set(weight_dict.keys()) == set(worker.model_runner.model.state_dict().keys()), "something wrong, model parameters key name mismatched"
    worker.model_runner.model.load_state_dict(weight_dict)

def debug_print():
    cprint(f"zhijiang, rank is {torch.distributed.get_rank()} done", "red", flush=True)