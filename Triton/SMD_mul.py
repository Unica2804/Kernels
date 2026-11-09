"""
Sparse Matrix - Dense Matrix multiplication in Triton.

    C = A @ B^T

    A: (M, K)  -- ~70% sparse (zeros injected)
    B: (N, K)  -- dense
    C: (M, N)

Both A and B are K-contiguous, so tiles load coalesced along K and the
B tile is transposed in-register for the dot product. The thread-block
launch order uses a 2D swizzle (tl.swizzle2d) to improve L2 locality.
"""

import triton
import triton.language as tl
import torch


@triton.jit
def spmm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_programs_m = tl.cdiv(M, BM)
    num_programs_n = tl.cdiv(N, BN)

    # 2D swizzle: column-major order within groups of GM rows of tiles.
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_programs_m, num_programs_n, GM)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    # (BM, BK) and (BN, BK) tiles, both K-contiguous.
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k_start in range(0, K, BK):
        k_mask = k_start + offs_k < K
        a_tile = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        b_tile = tl.load(
            b_ptrs,
            mask=(offs_n[:, None] < N) & k_mask[None, :],
            other=0.0,
        )

        # C[m, n] = sum_k A[m, k] * B[n, k]
        acc += tl.dot(a_tile, tl.trans(b_tile), input_precision="ieee")

        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def solve(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    nnz: int,
):
    if nnz == 0:
        C.zero_()
        return C

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    GROUP_M = 8

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    spmm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BM=BLOCK_M,
        BN=BLOCK_N,
        BK=BLOCK_K,
        GM=GROUP_M,
        num_warps=4,
        num_stages=3,
    )
    return C


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA GPU required for verification."

    test_cases = [
        (512, 512, 512),
        (1024, 768, 512),
        (137, 219, 73),
    ]

    for M, N, K in test_cases:
        print(f"Testing shape M={M}, N={N}, K={K}...")

        # A: (M, K) sparse, B: (N, K) dense, C: (M, N)
        A = torch.randn((M, K), device=device, dtype=torch.float32)
        B = torch.randn((N, K), device=device, dtype=torch.float32)
        C = torch.empty((M, N), device=device, dtype=torch.float32)

        # Inject 70% sparsity into A
        A[torch.rand((M, K), device=device) < 0.70] = 0.0
        nnz = int((A != 0).sum().item())

        # Ground truth: C = A @ B^T
        C_ref = torch.matmul(A, B.t())

        solve(A, B, C, M, N, K, nnz)

        torch.testing.assert_close(C, C_ref, atol=1e-3, rtol=1e-3)
        print(f"  -> Sparsity: {100.0 * (1.0 - nnz / (M * K)):.1f}% | NNZ: {nnz} | Passed!\n")

    print("All correctness verification checks passed successfully!")


if __name__ == "__main__":
    main()
