import torch
import triton
import triton.language as tl

@triton.jit
def combine(a1,x1,a2,x2):
    return a2*a1, a2*x1 + x2

@triton.jit
def linear_recurrence_kernel(
    a_ptr, x_ptr, h_ptr,
    B,L,
    stride_ab, stride_al,
    stride_xb, stride_xl,
    stride_hb, stride_hl,
    BLOCK_L : tl.constexpr,
):
    pid_b = tl.program_id(0)

    a_row_ptr = a_ptr + pid_b * stride_ab
    x_row_ptr = x_ptr + pid_b * stride_xb
    h_row_ptr = h_ptr + pid_b * stride_hb

    h_prev = 0.0

    for start_l in range(0, L, BLOCK_L):
        offs_l = start_l + tl.arange(0, BLOCK_L)
        mask = offs_l < L

        a_tile = tl.load(a_row_ptr + offs_l * stride_al, mask=mask, other=1.0)
        x_tile = tl.load(x_row_ptr + offs_l * stride_xl, mask=mask, other=0.0)

        is_first = tl.arange(0, BLOCK_L) == 0
        x_tile = tl.where(is_first, x_tile + a_tile * h_prev, x_tile)

        _, h_tile = tl.associative_scan((a_tile, x_tile), axis=0, combine_fn=combine)

        tl.store(h_row_ptr + offs_l * stride_hl, h_tile, mask=mask)

        last_valid_idx = min(BLOCK_L, L - start_l) - 1
        h_prev = tl.sum(tl.where(tl.arange(0, BLOCK_L) == last_valid_idx, h_tile, 0.0), axis=0)

def linear_recurrence(a, x):
    B, L = x.shape
    h = torch.empty_like(x)
    BLOCK_L = min(triton.next_power_of_2(L), 2048)
    BLOCK_L = max(BLOCK_L, 16)

    grid = (B,)
    linear_recurrence_kernel[grid](
        a, x, h,
        B, L,
        a.stride(0), a.stride(1),
        x.stride(0), x.stride(1),
        h.stride(0), h.stride(1),
        BLOCK_L=BLOCK_L,
        num_warps=4
    )
    return h

def main():
    B, L = 4, 8
    a = torch.rand(B, L, device='cuda')
    x = torch.rand(B, L, device='cuda')

    h = linear_recurrence(a, x)

    # Verify correctness with PyTorch
    h_ref = torch.empty_like(x)
    for b in range(B):
        h_ref[b, 0] = x[b, 0]
        for l in range(1, L):
            h_ref[b, l] = a[b, l] * h_ref[b, l - 1] + x[b, l]

    assert torch.allclose(h, h_ref), "Results do not match!"
    print("Results match with PyTorch implementation.")
if __name__ == "__main__":
    main()