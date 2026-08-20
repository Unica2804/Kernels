import torch
import triton
import triton.language as tl

configs = [
    triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=4),
]

@triton.autotune(configs=configs, key=['N'],reset_to_zero=['out_ptr'])
@triton.jit
def reduction(in_ptr,out_ptr,N,BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    offset_start = pid*BLOCK_SIZE
    stride = num_pid*BLOCK_SIZE
    acc = 0.0
    for block_offset in range(offset_start,N,stride):
        cols = block_offset+tl.arange(0,BLOCK_SIZE)
        mask = cols<N
        a = tl.load(in_ptr+cols,mask=mask,other=0.0).to(tl.float32)
        acc += tl.sum(a,axis=0)
    tl.atomic_add(out_ptr,acc)

def reduce(input: torch.Tensor)-> torch.Tensor:
    assert input.is_cuda, "Input tensor must be on GPU"
    N = input.numel()
    output = torch.zeros(1, device=input.device, dtype=torch.float32)
    grid = lambda meta: (min(1024,triton.cdiv(N,meta['BLOCK_SIZE'])),)
    reduction[grid](input,output,N)
    return output.to(input.dtype)

def benchmark():
    input = torch.randn(1<<24, device='cuda', dtype=torch.bfloat16)
    torch_op = input.sum()
    triton_op = reduce(input)
    assert torch.allclose(triton_op, torch_op, rtol=1e-3, atol=1e-3), "Results do not match!"
    triton_ms = triton.testing.do_bench(lambda: reduce(input))
    torch_ms = triton.testing.do_bench(lambda: input.sum())
    print(f"Triton time: {triton_ms:.2f} ms, Torch time: {torch_ms:.2f} ms")
    print(f"Triton config: {reduction.best_config}")

if __name__ == "__main__":
    benchmark()