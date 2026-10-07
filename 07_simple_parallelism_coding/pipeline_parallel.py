import os
import math
from typing import Callable

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank: int, world_size: int):
    """
    Initializes the distributed environment (called at start of process).
    """
    # Specify where master lives (rank 0), used to coordinate (actual data goes through NCCL)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    
    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    """
    Cleans up the distributed environment (called at end of process).
    """
    dist.destroy_process_group()


def cuda_if_available(index: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")


def get_init_params(num_inputs: int, num_outputs: int, rank: int) -> torch.nn.Parameter:
    """
    Creates parameters and puts them on the `rank`-th GPU.
    """
    torch.random.manual_seed(0)
    return torch.nn.Parameter(torch.randn(num_inputs, num_outputs, device=cuda_if_available(rank)) / math.sqrt(num_outputs))


def int_divide(a: int, b: int) -> int:
    assert a % b == 0
    return a // b


def summarize_tensor(tensor: torch.tensor) -> str:
    return "x".join(map(str, tensor.shape)) + "[" + str(round(tensor.view(-1)[0].item(), 4)) + "...]"


def pipeline_parallelism_main(rank: int, world_size: int, data: torch.tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)
    
    batch_size = data.size(0)
    num_dim = data.size(1)
    data = data.to(cuda_if_available(rank))

    # Split up layers (each rank gets a subset of layers)
    local_num_layers = int_divide(num_layers, world_size)
    local_params = [get_init_params(num_dim, num_dim, rank) for layer in range(local_num_layers)]
    
    # Forward pass
    # Break up into micro batches to minimize the bubble
    micro_batch_size = int_divide(batch_size, num_micro_batches)
    if rank == 0:
        # The data
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # Allocate memory for activations
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=cuda_if_available(rank)) for _ in range(num_micro_batches)]
    
    for x in micro_batches:
        # Get activations from previous rank
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)
        # Compute layers assigned to this rank
        for param in local_params:
            x = x @ param
            x = F.gelu(x)
        # Send to the next rank
        if rank + 1 < world_size:
            print(f"[pipeline_parallelism] Rank {rank}: sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)
            dist.send(tensor=x, dst=rank + 1)
    
    # Not handled: overlapping communication/computation to eliminate pipeline bubbles
    
    cleanup()


def generate_sample_data():
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)
    return data


def spawn(func: Callable, world_size: int, *args, **kwargs):
    """
    Launches `world_size` processes that each calls `func` on world_size, args, kwargs.
    """
    args = (world_size,) + args + tuple(kwargs.values())
    mp.spawn(func, args=args, nprocs=world_size, join=True)


if __name__ == "__main__":
    data = generate_sample_data()
    spawn(pipeline_parallelism_main, world_size=2, data=data, num_layers=4, num_micro_batches=4)