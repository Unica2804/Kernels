import torch
import triton
import triton.language as tl


@triton.jit
def geglu_kernel(input, output, N, BLOCK_SIZE: tl.constexpr):
    pid=tl.program_id(0)
    offsets=pid*BLOCK_SIZE+tl.arange(0,BLOCK_SIZE)
    half_n = N//2
    mask = offsets<half_n
    val = tl.load(input+offsets,mask=mask)
    gate = tl.load(input+offsets+half_n,mask=mask)


    gate = 0.5 * gate * (1+tl.math.erf(gate*0.70710678))
    out = val*gate
    tl.store(output+offsets, out, mask=mask)


def geglu(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    geglu_kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)

def main():
    N = 1 << 20
    input = torch.randn(N, device='cuda', dtype=torch.float32)
    output = torch.empty(N//2, device='cuda', dtype=torch.float32)

    geglu(input, output, N)

    gate = input[N//2:]
    value = input[:N//2]
    torch_output = value * (0.5 * gate * (1 + torch.erf(gate * 0.70710678)))

    assert torch.allclose(output, torch_output, atol=1e-6), "Triton and PyTorch results do not match!"
    print("Correctness test passed!")

    triton_ms = triton.testing.do_bench(lambda: geglu(input, output, N))
    print(f"Triton GeGLU kernel time: {triton_ms:.2f} ms")

if __name__ == "__main__":
    main()