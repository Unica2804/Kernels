import triton
import triton.language as tl
import torch

@triton.jit
def leaky_relu_kernel(input_ptr, output_ptr, N, slope, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(input_ptr + offset, mask=mask)
    y = tl.where(x >= 0, x, slope * x)
    tl.store(output_ptr + offset, y, mask=mask)

def leaky_relu(input: torch.Tensor, output: torch.Tensor, slope: float):
    N = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    leaky_relu_kernel[grid](input, output, N, slope, BLOCK_SIZE)
    return output

x= torch.randn(1 << 20, device='cuda', dtype=torch.float32)
y = torch.empty_like(x)
slope = 0.01
triton_op = leaky_relu(x, y, slope)
torch_op = torch.where(x >= 0, x, slope * x)
assert torch.allclose(triton_op, torch_op)
print("Correctness test passed!")
torch_ms = triton.testing.do_bench(lambda: leaky_relu(x, y, slope))
print(f"Triton Leaky ReLU kernel time: {torch_ms:.2f} ms")