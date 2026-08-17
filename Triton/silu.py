import triton
import triton.language as tl
import torch

@triton.jit
def Silu_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X_ptr+offsets, mask=mask)
    y = x / (1 + tl.exp(-x))
    tl.store(Y_ptr+offsets, y, mask=mask)

def silu(X: torch.Tensor) -> torch.Tensor:
    Y = torch.empty_like(X)
    N = X.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    Silu_kernel[grid](X, Y, N, BLOCK_SIZE)
    return Y

N = 1 << 20
input = torch.randn(N, device='cuda', dtype=torch.float32)

triton_op = silu(input)
torch_op = torch.nn.functional.silu(input)

assert torch.allclose(triton_op, torch_op, atol=1e-6), "Triton and PyTorch results do not match!"
print("Correctness test passed!")
triton_ms = triton.testing.do_bench(lambda: silu(input))
print(f"Triton SiLU kernel time: {triton_ms:.2f} ms")