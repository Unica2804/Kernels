import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid*BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(input+offset, mask=mask)
    y = max(x,0)
    tl.store(output+offset,y,mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_kernel[grid](input, output, N, BLOCK_SIZE)

def main():
    N = 1 << 20
    input = torch.randn(N, device='cuda', dtype=torch.float32)
    output = torch.empty_like(input)
    solve(input, output, N)
    triton_ms = triton.testing.do_bench(lambda: solve(input, output, N))
    assert torch.allclose(output, torch.clamp(input, min=0))
    print("Correctness test passed!")
    print(f"Triton ReLU kernel time: {triton_ms:.2f} ms")

if __name__ == "__main__":
    main()