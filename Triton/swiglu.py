import triton
import triton.language as tl
import torch

@triton.jit
def swiglu_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N//2
    gate = tl.load(X_ptr+offsets, mask=mask)
    value = tl.load(X_ptr+offsets+N//2, mask=mask)
    gate = gate * tl.sigmoid(gate)
    y = gate * value
    tl.store(Y_ptr+offsets, y, mask=mask)

def swiglu(X: torch.Tensor) -> torch.Tensor:
    N = X.numel()
    Y = torch.empty_like(X[:N//2])
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N//2, BLOCK_SIZE),)
    swiglu_kernel[grid](X, Y, N, BLOCK_SIZE)
    return Y

N = 1 << 20
input = torch.randn(N, device='cuda', dtype=torch.float32)
triton_op = swiglu(input)
gate = input[:N//2]
value = input[N//2:]
torch_op = gate * torch.sigmoid(gate) * value
assert torch.allclose(triton_op, torch_op, atol=1e-6), "Triton and PyTorch results do not match!"
print("Correctness test passed!")
triton_ms = triton.testing.do_bench(lambda: swiglu(input))
print(f"Triton SwiGLU kernel time: {triton_ms:.2f} ms")