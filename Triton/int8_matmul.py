import triton as tl
import torch

import torch
import triton
import triton.language as tl

@triton.jit
def int8_mul(a, b, c, M, N, K,
             stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
             scale_A, scale_B, scale_C,
             zero_point_A, zero_point_B, zero_point_C,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    row_acc = tl.zeros((BLOCK_M,), dtype=tl.int32)
    col_acc = tl.zeros((BLOCK_N,), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_a = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        mask_b = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        a_tile = tl.load(a_ptrs, mask=mask_a, other=0)
        b_tile = tl.load(b_ptrs, mask=mask_b, other=0)

        acc = tl.dot(a_tile, b_tile, acc, out_dtype=tl.int32)

        row_acc += tl.sum(tl.where(mask_a, a_tile.to(tl.int32), 0), axis=1)
        col_acc += tl.sum(tl.where(mask_b, b_tile.to(tl.int32), 0), axis=0)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    correction = (
        -zero_point_B * row_acc[:, None]
        - zero_point_A * col_acc[None, :]
        + K * zero_point_A * zero_point_B
    )
    acc_corrected = acc + correction

    effective_s = scale_A * scale_B / scale_C
    val_f32 = acc_corrected.to(tl.float32) * effective_s
    val_int = tl.extra.cuda.libdevice.nearbyint(val_f32) + zero_point_C
    val_clamp = tl.clamp(val_int, -128.0, 127.0)

    c_ptrs = c + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, val_clamp.to(tl.int8), mask=mask_c)

# a, b, c are tensors on the GPU
def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    scale_A: float,
    scale_B: float,
    scale_C: float,
    zero_point_A: int,
    zero_point_B: int,
    zero_point_C: int,
):
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    GROUP_SIZE = 8
    stride_am,stride_ak = a.stride(0),a.stride(1)
    stride_bk,stride_bn = b.stride(0),b.stride(1)
    stride_cm,stride_cn = c.stride(0),c.stride(1)

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    int8_mul[grid](
        a,b,c,
        M,N,K,
        stride_am,stride_ak,
        stride_bk,stride_bn,
        stride_cm,stride_cn,
        scale_A,scale_B,scale_C,
        zero_point_A,zero_point_B,zero_point_C,
        BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K,GROUP_SIZE
    )


def reference(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_A: float,
    scale_B: float,
    scale_C: float,
    zero_point_A: int,
    zero_point_B: int,
    zero_point_C: int,
) -> torch.Tensor:
    """Compute the quantized matmul result with PyTorch for test comparison."""
    acc = (a.to(torch.int32) - zero_point_A) @ (b.to(torch.int32) - zero_point_B)
    result = torch.round(
        acc.to(torch.float32) * (scale_A * scale_B / scale_C)
    ) + zero_point_C
    return result.clamp(-128, 127).to(torch.int8)


def test_case(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_A: float,
    scale_B: float,
    scale_C: float,
    zero_point_A: int,
    zero_point_B: int,
    zero_point_C: int,
) -> None:
    m, k = a.shape
    b_k, n = b.shape
    assert k == b_k

    output = torch.empty((m, n), device="cuda", dtype=torch.int8)
    solve(
        a.cuda(),
        b.cuda(),
        output,
        m,
        n,
        k,
        scale_A,
        scale_B,
        scale_C,
        zero_point_A,
        zero_point_B,
        zero_point_C,
    )
    torch.cuda.synchronize()

    expected = reference(
        a,
        b,
        scale_A,
        scale_B,
        scale_C,
        zero_point_A,
        zero_point_B,
        zero_point_C,
    )
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    print(f"{name}: passed")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("int8_matmul tests require a CUDA device")

    test_case(
        "symmetric non-multiple dimensions",
        torch.tensor(
            [[1, 2, -3, 4, 5], [-6, 7, 8, -9, 10], [11, -12, 13, 14, -15]],
            dtype=torch.int8,
        ),
        torch.tensor(
            [
                [2, -3, 4, 5],
                [-6, 7, -8, 9],
                [10, 11, -12, 13],
                [-14, 15, 16, -17],
                [18, -19, 20, 21],
            ],
            dtype=torch.int8,
        ),
        1.0,
        1.0,
        1.0,
        0,
        0,
        0,
    )
    test_case(
        "asymmetric zero points and requantization",
        torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.int8),
        torch.tensor(
            [[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]],
            dtype=torch.int8,
        ),
        0.5,
        0.25,
        1.0,
        5,
        7,
        -3,
    )

    generator = torch.Generator().manual_seed(0)
    test_case(
        "multiple K tiles and output clamping",
        torch.randint(-128, 127, (65, 65), generator=generator, dtype=torch.int8),
        torch.randint(-128, 127, (65, 67), generator=generator, dtype=torch.int8),
        1.0,
        1.0,
        0.25,
        0,
        0,
        0,
    )


if __name__ == "__main__":
    main()
