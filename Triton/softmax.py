import torch
import triton
import triton.language as tl

config = [
    triton.Config({'BLOCK_SIZE': 8192}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE': 8192}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=4),    
]

@triton.autotune(
    configs=config,
    key=['N'],
    reset_to_zero=['output'],
)
@triton.jit
def softmax_1D_kernel(input, output, N, BLOCK_SIZE: tl.constexpr):
    #calculate global max(x)
    global_max = float("-inf")
    for col_start in range(0,N,BLOCK_SIZE):
        offset = col_start + tl.arange(0,BLOCK_SIZE)
        mask = offset<N
        x = tl.load(input+offset,mask=mask,other=float("-inf")).to(tl.float32)
        max = tl.max(x,axis=0)
        global_max = tl.maximum(max,global_max)
    #calculate global sum(stable(exp(x)))
    global_sum = 0.0
    for col_start in range(0,N,BLOCK_SIZE):
        offset = col_start + tl.arange(0,BLOCK_SIZE)
        mask = offset<N
        x = tl.load(input+offset,mask=mask,other=float("-inf")).to(tl.float32)
        stable_exp = tl.exp(x-global_max)
        sum = tl.sum(stable_exp,axis=0)
        global_sum += sum
    #calculate softmax
    for col_start in range(0,N,BLOCK_SIZE):
        offset = col_start + tl.arange(0,BLOCK_SIZE)
        mask = offset<N
        x = tl.load(input+offset,mask=mask,other=float("-inf")).to(tl.float32)
        stable_exp = tl.exp(x-global_max)
        y=stable_exp/global_sum
        tl.store(output+offset,y,mask=mask)

@triton.autotune(
    configs=config,
    key=['N'],
    reset_to_zero=['output'],
)
@triton.jit
def online_softmax(input, output, N, BLOCK_SIZE: tl.constexpr):
    # pass 1 calculate global max(x) and global sum(stable(exp(x)))
    global_max = float("-inf")
    global_sum = 0.0
    for col_start in range(0,N,BLOCK_SIZE):
        offset = col_start + tl.arange(0,BLOCK_SIZE)
        mask = offset<N
        x = tl.load(input+offset,mask=mask,other=float("-inf")).to(tl.float32)
        cur_max = tl.max(x,axis=0)
        new_max = tl.maximum(cur_max,global_max)
        global_sum = global_sum * tl.exp(global_max-new_max) + tl.sum(tl.exp(x-new_max),axis=0)
        global_max = new_max
    # pass 2 calculate softmax
    for col_start in range(0,N,BLOCK_SIZE):
        offset = col_start + tl.arange(0,BLOCK_SIZE)
        mask = offset<N
        x = tl.load(input+offset,mask=mask,other=float("-inf")).to(tl.float32)
        stable_exp = tl.exp(x-global_max)
        y=stable_exp/global_sum
        tl.store(output+offset,y,mask=mask)

def solve_1D(input: torch.Tensor, output: torch.Tensor, N: int):
    grid = (1,)
    softmax_1D_kernel[grid](
        input,output,N
    )
    return output

def online_solve(input: torch.Tensor, output: torch.Tensor, N: int):
    grid = (1,)
    online_softmax[grid](
        input,output,N
    )
    return output

def benchmark(N):
    torch.backends.cuda.matmul.allow_tf32 = True
    dtypes = {
        "float32": (torch.float32, 4),
        "bfloat16": (torch.bfloat16, 2),
    }
    for dtype_name, (dtype, elem_size) in dtypes.items():
        print(f"\n=== dtype: {dtype_name} ===")
        inp = torch.randn(N,device='cuda',dtype=dtype)
        ref = torch.nn.functional.softmax(inp.float(),dim=0)
        
        out_1D = torch.zeros_like(inp)
        out_online = torch.zeros_like(inp)
        
        res_1D = solve_1D(inp,out_1D.clone(),N)
        res_online = online_solve(inp,out_online.clone(),N)
        
        print(f"Max Error (1D): {torch.max(torch.abs(ref - res_1D.float()))}")
        print(f"Max Error (Online): {torch.max(torch.abs(ref - res_online.float()))}")
        total_bytes = 2*N*elem_size
        torch_ms = triton.testing.do_bench(lambda: torch.nn.functional.softmax(inp,dim=0))
        
        ms_1D = triton.testing.do_bench(lambda: solve_1D(inp,out_1D,N))
        ms_online = triton.testing.do_bench(lambda: online_solve(inp,out_online,N))
        print(f"Bandwidth (Torch): {total_bytes/(torch_ms*1e-3)/1e9:.2f} GB/s")
        
        print(f"Bandwidth (1D): {total_bytes/(ms_1D*1e-3)/1e9:.2f} GB/s")
        print(f"Bandwidth (Online): {total_bytes/(ms_online*1e-3)/1e9:.2f} GB/s")
if __name__ == "__main__":
    benchmark(5000000)