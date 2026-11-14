"""
Weight-Activation 4-bit * 16-bit multiplication in Triton.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def w4a16_kernel(
    x_ptr, w_ptr, scales_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_sn, stride_sk,
    stride_ym, stride_yn,
    group_size,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GSZ: tl.constexpr,
) :
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2D swizzle: column-major order within groups of GSZ rows of tiles.
    num_programs_m = tl.cdiv(M, BM)
    num_programs_n = tl.cdiv(N, BN)
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_programs_m, num_programs_n, GSZ)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k_packed = tl.arange(0, BK//2)

    #base pointers
    x_base = x_ptr + offs_m[:, None] * stride_xm + (offs_k_packed[None, :] * 2) * stride_xk
    w_base = w_ptr + offs_n[:, None] * stride_wn + offs_k_packed[None, :] * stride_wk
    
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k_start in range(0, K, BK):
        mask_m = offs_m[:, None] < M
        mask_n = offs_n[:, None] < N  # (BN, 1) so it broadcasts against (1, BK//2)

        #split k in two halves, each half is 4-bit
        k_even = k_start + offs_k_packed * 2
        k_odd = k_even + 1

        mask_x_even = mask_m & (k_even[None, :] < K)
        mask_x_odd = mask_m & (k_odd[None, :] < K)
        mask_w = mask_n & (offs_k_packed[None, :] < (tl.cdiv(K, 2) - k_start // 2))

        # x_base already includes offs_m * stride_xm; just advance along K
        x_even_tile = tl.load(x_base + k_start * stride_xk, mask=mask_x_even, other=0.0)
        x_odd_tile = tl.load(x_base + (k_start + 1) * stride_xk, mask=mask_x_odd, other=0.0)

        packed_w_tile = tl.load(w_base,mask=mask_w, other=0)

        # unpack 4-bit weights to signed 4-bit integers
        w_even_u4 = (packed_w_tile >> 4) & 0x0F
        w_odd_u4 = packed_w_tile & 0x0F

        w_even_s4 = w_even_u4.to(tl.float32) - 8.0
        w_odd_s4 = w_odd_u4.to(tl.float32) - 8.0

        # Load scales for dequantization
        scales_group_even = k_even // group_size
        scales_group_odd = k_odd // group_size

        scale_even_ptrs = scales_ptr + offs_n[:, None] * stride_sn + scales_group_even[None, :] * stride_sk
        scale_odd_ptrs = scales_ptr + offs_n[:, None] * stride_sn + scales_group_odd[None, :] * stride_sk

        scale_even_tile = tl.load(scale_even_ptrs, mask=mask_w, other=0.0)
        scale_odd_tile = tl.load(scale_odd_ptrs, mask=mask_w, other=0.0)

        # Dequantize weights
        w_even_f16 = (w_even_s4 * scale_even_tile).to(tl.float16)
        w_odd_f16 = (w_odd_s4 * scale_odd_tile).to(tl.float16)

        # Compute the dot product for the current tile
        acc += tl.dot(x_even_tile, tl.trans(w_even_f16))
        acc += tl.dot(x_odd_tile, tl.trans(w_odd_f16))

        w_base += (BK // 2) * stride_wk
    
    y_ptrs = y_ptr + offs_m[:,None]*stride_ym + offs_n[None,:]*stride_yn
    mask_y = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(tl.float16), mask=mask_y)

def w4a16_mul(
    X: torch.Tensor,
    W: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128
):  
    M, K = X.shape
    N = W.shape[0]

    Y = torch.empty((M, N), device=X.device, dtype=torch.float16)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_SIZE = 8

    assert BLOCK_K % 2 == 0, "BLOCK_K must be even for 4-bit weights"

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    w4a16_kernel[grid](
        X, W, scales, Y,
        M, N, K,
        X.stride(0), X.stride(1),
        W.stride(0), W.stride(1),
        scales.stride(0), scales.stride(1),
        Y.stride(0), Y.stride(1),
        group_size,
        BM=BLOCK_M,
        BN=BLOCK_N,
        BK=BLOCK_K,
        GSZ=GROUP_SIZE
    )
    return Y

def weight_pack (
    weights: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128
):
    """
    Pack 4-bit weights and scales for Triton kernel.

    Args:
        weights (torch.Tensor): The weight tensor of shape (N, K) with dtype torch.float16.
        scales (torch.Tensor): The scale tensor of shape (N, K//group_size) with dtype torch.float16.
        group_size (int): The size of the group for quantization.
    Returns:
        packed_weights (torch.Tensor): The packed weight tensor of shape (N, K//2) with dtype torch.int8.
    """
    _ , K = weights.shape
    assert K % 2 == 0, "K must be even for 4-bit weights"

    # Quantize weights to signed 4-bit integers
    quantized_weights = torch.clamp((weights / scales[:, :K // group_size].repeat_interleave(group_size, dim=1)).round(), -8, 7)

    # Pack two 4-bit weights into one byte.
    # Store offset-binary (q + 8) nibbles so the kernel's `nibble - 8.0` decode recovers q.
    offset_weights = (quantized_weights + 8).to(torch.int16)  # values in [0, 15]
    packed_weights = (offset_weights[:, ::2] << 4) | offset_weights[:, 1::2]

    return packed_weights.to(torch.int8)

def main():
    M, K, N = 128, 256, 64
    group_size = 128

    X = torch.randn((M, K), device='cuda', dtype=torch.float16)
    W = torch.randn((N, K), device='cuda', dtype=torch.float16)
    scales = torch.randn((N, K // group_size), device='cuda', dtype=torch.float16)

    packed_W = weight_pack(W, scales, group_size)
    Y = w4a16_mul(X, packed_W, scales, group_size)

    # Reference: unpack the packed weights the same way the kernel does
    # (offset-binary nibbles, high nibble = even k), dequantize, then matmul.
    N_rows, K = packed_W.shape
    pw = packed_W.to(torch.int16) 
    s4 = torch.empty((N_rows, K * 2), dtype=torch.float16, device=W.device)
    s4[:, 0::2] = ((pw >> 4) & 0x0F).to(torch.float16) - 8.0
    s4[:, 1::2] = (pw & 0x0F).to(torch.float16) - 8.0
    dequant_W = s4 * scales.repeat_interleave(group_size, dim=1)
    Y_ref = torch.matmul(X, dequant_W.t())

    torch.testing.assert_close(Y, Y_ref, rtol=1e-2, atol=1e-2)
    print("correctness passed!")
    print("max |Y - Y_ref|          :", (Y - Y_ref).abs().max().item())
    print("max |Y - full precision| :", (Y - torch.matmul(X, W.t())).abs().max().item(), " (expected: quantization error)")

if __name__ == "__main__":
    main()