import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert x.device == DEVICE and y.device == DEVICE and out.device == DEVICE

    N = x.numel()
    BLOCK_SIZE = 1024
    grid_size = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_size,)
    add_kernel[grid](x, y, out, N, BLOCK_SIZE)
    return out

torch.manual_seed(0)
size = 1198432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is 'f'{torch.max(torch.abs(output_torch - output_triton))}')
