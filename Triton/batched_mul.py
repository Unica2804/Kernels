import torch
import triton
import triton.language as tl

@triton.jit
def batched_mul(
    a_ptr,b_ptr,c_ptr,
    m,n,k,
    stride_ab,stride_am,stride_ak,
    stride_bb,stride_bk,stride_bn,
    stride_cb,stride_cm,stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_spatial = tl.program_id(0)
    batch_idx = tl.program_id(1)

    #L2 swizzling
    num_pid_m = tl.cdiv(m, BLOCK_M)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid_spatial // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid_spatial % group_size_m)
    pid_n = (pid_spatial % num_pid_in_group) // group_size_m

    # loading pointers
    idx_a = a_ptr+batch_idx*stride_ab
    idx_b = b_ptr+batch_idx*stride_bb
    idx_c = c_ptr+batch_idx*stride_cb

    a_blk = tl.make_block_ptr(
        idx_a,
        (m,k),
        (stride_am,stride_ak),
        (pid_m*BLOCK_M,0),
        (BLOCK_M,BLOCK_K),
        (1,0),
    )
    b_blk = tl.make_block_ptr(
        idx_b,
        (k,n),
        (stride_bk,stride_bn),
        (0,pid_n*BLOCK_N),
        (BLOCK_K,BLOCK_N),
        (1,0),
    )
    c_blk = tl.make_block_ptr(
        idx_c,
        (m,n),
        (stride_cm,stride_cn),
        (pid_m*BLOCK_M,pid_n*BLOCK_N),
        (BLOCK_M,BLOCK_N),
        (1,0),
    )
    acc = tl.zeros((BLOCK_M,BLOCK_N),dtype=tl.float32)

    for _ in range(0,k,BLOCK_K):
        a_tile = tl.load(a_blk,boundary_check=(0, 1), padding_option="zero")
        b_tile = tl.load(b_blk,boundary_check=(0, 1), padding_option="zero")
        acc+=tl.dot(a_tile,b_tile)
        a_blk = a_blk.advance((0,BLOCK_K))
        b_blk = b_blk.advance((BLOCK_K,0))
    tl.store(c_blk,acc.to(c_ptr.dtype.element_ty),boundary_check=(0,1))


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, BATCH: int, M: int, N: int, K: int):
    
    if a.ndim == 3:
        stride_ab, stride_am, stride_ak = a.stride(0), a.stride(1), a.stride(2)
        stride_bb, stride_bk, stride_bn = b.stride(0), b.stride(1), b.stride(2)
        stride_cb, stride_cm, stride_cn = c.stride(0), c.stride(1), c.stride(2)
    else:
        # 1D flattened contiguous row-major strides:
        # A: shape (BATCH, M, K) -> strides (M*K, K, 1)
        stride_ab, stride_am, stride_ak = M * K, K, 1
        # B: shape (BATCH, K, N) -> strides (K*N, N, 1)
        stride_bb, stride_bk, stride_bn = K * N, N, 1
        # C: shape (BATCH, M, N) -> strides (M*N, N, 1)
        stride_cb, stride_cm, stride_cn = M * N, N, 1

    BLOCK_M = 128
    BLOCK_K = 64
    BLOCK_N = 128
    GROUP_M = 8
    grid = (triton.cdiv(M,BLOCK_M)*triton.cdiv(N,BLOCK_N),BATCH,)
    batched_mul[grid] (
        a,b,c,
        M,N,K,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_M,BLOCK_N,BLOCK_K,GROUP_M,
    )
    return c

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    BATCH = 4
    M = 2048
    N = 1024
    K = 2048

    a = torch.randn(BATCH, M, K, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(BATCH, K, N, device='cuda', dtype=torch.bfloat16)
    c = torch.empty(BATCH, M, N, device='cuda', dtype=torch.bfloat16)

    c_triton = solve(a, b, c, BATCH, M, N, K)
    c_torch = torch.bmm(a.to(torch.float32), b.to(torch.float32)).to(torch.bfloat16)

    print("Difference:", torch.max(torch.abs(c_triton - c_torch)))

    triton_ms = triton.testing.do_bench(lambda: solve(a, b, c, BATCH, M, N, K))
    torch_ms = triton.testing.do_bench(lambda: torch.bmm(a.to(torch.float32), b.to(torch.float32)).to(torch.bfloat16))
    print(f"Triton time: {triton_ms:.2f} ms")
    print(f"Torch time: {torch_ms:.2f} ms")

if __name__ == "__main__":
    main()