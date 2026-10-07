import torch
import torch.nn as nn
import einops
from jaxtyping import Float


print(torch.cuda.is_available())
num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    properties = torch.cuda.get_device_properties(i)


x = torch.empty(4, 8)
nn.init.trunc_normal_(x, mean=0, std=1, a=-2, b=2)
print(x)


# jaxtyping
x = torch.ones(2, 2, 1, 3)
x: Float[torch.Tensor, "batch seq head hidden"] = torch.ones(2, 2, 1, 3)

# einsum
x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)
y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4)
# old way
z = x @ y.transpose(-1, -2)
# new way
z = einops.einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
# or
z = einops.einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")

# reduce
x: Float[torch.Tensor, "batch seq hidden"] = torch.ones(2, 3, 4)
# old way
y = x.mean(dim=-1)
# new way
y = einops.reduce(x, "... hidden -> ...", "mean")

# rearrange
x: Float[torch.Tensor, "batch seq total_hidden"] = torch.ones(2, 3, 8)
w: Float[torch.Tensor, "hidden1 hidden2"] = torch.ones(4, 4)
x = einops.rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)
x = einops.einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")
x = einops.rearrange(x, "... heads hidden2 -> ... (heads hidden2)")


import random

random.seed(42)
print(random.random())