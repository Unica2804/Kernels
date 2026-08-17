import torch
import triton
import triton.language as tl

@triton.jit
def transpose_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )

    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(N, M),
        strides=(stride_ym, stride_yn),  
        offsets=(pid_n * BLOCK_SIZE_N, pid_m * BLOCK_SIZE_M),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_M),
        order=(1, 0),
    )

    x_tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
    y_tile = tl.trans(x_tile)
    tl.store(y_block_ptr, y_tile, boundary_check=(0, 1))


def transpose(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda, "Input tensors must be on CUDA"
    assert x.dtype == y.dtype, "Input and output tensors must have the same dtype"
    assert x.shape[0] == y.shape[1] and x.shape[1] == y.shape[0], "Shape mismatch"

    M, N = x.shape
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    transpose_kernel[grid](
        x, y,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=4,
    )
    return y


def benchmark_transpose(N: int = 4096):
    device = "cuda"
    x = torch.randn((N, N), device=device, dtype=torch.float32)
    y_triton = torch.empty((N, N), device=device, dtype=torch.float32)

    transpose(x, y_triton)
    torch.testing.assert_close(y_triton, x.t().contiguous())
    print("Correctness check passed!")

    triton_ms = triton.testing.do_bench(lambda: transpose(x, y_triton))
    torch_ms = triton.testing.do_bench(lambda: x.t().contiguous())

    gb_transferred = (2 * N * N * 4) / 1e9
    triton_gbps = gb_transferred / (triton_ms * 1e-3)
    torch_gbps = gb_transferred / (torch_ms * 1e-3)

    print(f"\nMatrix Size: {N}x{N}")
    print(f"{'Implementation':<18} | {'Time (ms)':<10} | {'Bandwidth (GB/s)':<18}")
    print("-" * 52)
    print(f"{'Triton':<18} | {triton_ms:<10.3f} | {triton_gbps:<18.2f}")
    print(f"{'PyTorch':<18} | {torch_ms:<10.3f} | {torch_gbps:<18.2f}")


if __name__ == "__main__":
    benchmark_transpose(4096)