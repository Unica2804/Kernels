import torch
import triton
import triton.language as tl
import triton.testing

autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4}, num_stages=3, num_warps=8),
]

@triton.autotune(
    configs=autotune_configs,
    key=['M', 'N', 'K'],
    reset_to_zero=['c_ptr'],
)

@triton.jit
def matrix_multiplication_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bn, stride_bk,
    stride_cm, stride_ck,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # --- Boiler plate Block Swizzling Code ---
    pid = tl.program_id(axis=0)
    # num tiles in each dimension
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    # num groups in the M dimension
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    # grp size usually 2,4 or 8, but can be smaller at the end of the grid
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # tile indices
    pid_m = first_pid_m + (pid % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m
    # --- End Boiler plate Block Swizzling Code ---

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, N),
        strides=(stride_am, stride_an),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        offsets=(0, pid_k * BLOCK_SIZE_K),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(1, 0),
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, K),
        strides=(stride_cm, stride_ck),
        offsets=(pid_m * BLOCK_SIZE_M, pid_k * BLOCK_SIZE_K),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for _ in range(0, N, BLOCK_SIZE_N):
        a_tile = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b_tile = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        accumulator += tl.dot(a_tile, b_tile, allow_tf32=True)
        a_block_ptr = a_block_ptr.advance((0, BLOCK_SIZE_N))
        b_block_ptr = b_block_ptr.advance((BLOCK_SIZE_N, 0))

    c_tile = accumulator.to(tl.bfloat16)
    tl.store(c_block_ptr, c_tile, boundary_check=(0, 1))


def triton_mul(a: torch.Tensor, b: torch.Tensor, c:torch.Tensor):
    stride_am, stride_an = a.stride(0), a.stride(1)
    stride_bn, stride_bk = b.stride(0), b.stride(1)
    stride_cm, stride_ck = c.stride(0), c.stride(1)
    M, N = a.shape
    N_b, K = b.shape
    assert N == N_b, "Inner dimensions must match for matrix multiplication."
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_K']),)

    matrix_multiplication_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
    )
    return c


def benchmark_all(N=2048):
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    a_int = torch.randint(-2, 3, (N, N), device=device, dtype=torch.bfloat16)
    b_int = torch.randint(-2, 3, (N, N), device=device, dtype=torch.bfloat16)
    c_triton = torch.empty((N, N), device=device, dtype=torch.bfloat16)
    torch_c = torch.matmul(a_int, b_int)

    c_triton = triton_mul(a_int, b_int, c_triton)

    torch.testing.assert_close(c_triton, torch_c, rtol=0, atol=0) 
    print("Correctness tests passed (Exact match on integer tensors)!\n")

    a = torch.randn((N, N), device=device, dtype=torch.bfloat16) 
    b = torch.randn((N, N), device=device, dtype=torch.bfloat16) 
    c_triton = torch.empty((N, N), device=device, dtype=torch.bfloat16)

    triton_ms = triton.testing.do_bench(lambda: triton_mul(a, b, c_triton)) 
    torch_ms = triton.testing.do_bench(lambda: torch.matmul(a, b)) 

    flops = 2.0 * (N**3)

    triton_tflops = (flops / (triton_ms * 1e-3)) / 1e12 
    torch_tflops = (flops / (torch_ms * 1e-3)) / 1e12 

    # Print Formatted Table
    print(f"Benchmark Results (Matrix size: {N}x{N}):")
    print(f"{'Implementation':<24} | {'Time (ms)':<10} | {'Throughput (TFLOPS)':<18}")
    print("-" * 60)
    print(f"{'Triton (Tiled + Dot)':<24} | {triton_ms:<10.3f} | {triton_tflops:<18.2f}")
    print(f"{'PyTorch (cuBLAS)':<24} | {torch_ms:<10.3f} | {torch_tflops:<18.2f}")
    print(f"Best config for N={N}: {matrix_multiplication_kernel.best_config}\n")

if __name__ == "__main__":
    benchmark_all(N=2048)
    benchmark_all(N=4096)