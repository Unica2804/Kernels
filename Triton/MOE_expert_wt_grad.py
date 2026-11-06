import triton
from triton import language as tl
import torch

@triton.jit
def MOE_grad(
    x_ptr,
    dy_ptr,
    dw_ptr,
    group_offsets_ptr,
    G:tl.constexpr,
    M_total:tl.constexpr,
    K:tl.constexpr,
    N:tl.constexpr,
    stride_xm, stride_xk,
    stride_dym, stride_dyn,
    stride_dwg,stride_dwk, stride_dwn,
    BM:tl.constexpr,
    BN:tl.constexpr,
    BK:tl.constexpr,
):
    pid_g = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_n = tl.program_id(2)

    # load slice boudaries for the current group
    s = tl.load(group_offsets_ptr + pid_g, mask=pid_g < G, other=0)
    e = tl.load(group_offsets_ptr + pid_g + 1, mask=pid_g < G, other=0)

    acc = tl.zeros((BK, BN), dtype=tl.float32)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_n = pid_n * BN + tl.arange(0, BN)

    for m_start in range(s,e,BM):
        offs_m = m_start + tl.arange(0, BM)

        x_ptrs = x_ptr + (offs_k[:, None]*stride_xk + offs_m[None, :]*stride_xm)
        x_mask = (offs_k[:, None] < K) & (offs_m[None, :] < e)
        x_t = tl.load(x_ptrs, mask=x_mask, other=0.0)

        dy_ptrs = dy_ptr + (offs_m[:, None]*stride_dym + offs_n[None, :]*stride_dyn)
        dy_mask = (offs_m[:, None] < e) & (offs_n[None, :] < N)
        dy = tl.load(dy_ptrs, mask=dy_mask, other=0.0)

        acc += tl.dot(x_t, dy)

    # store results
    dw_ptrs = dw_ptr + pid_g*stride_dwg + (offs_k[:, None]*stride_dwk + offs_n[None, :]*stride_dwn)
    dw_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    tl.store(dw_ptrs, acc, mask=dw_mask)


def moe_expert_weight_grad(x, dy, group_offsets, G):
    """
    x: (M_total, K)
    dy: (M_total, N)
    group_offsets: (G + 1,)
    returns: dw of shape (G, K, N)
    """
    M_total, K = x.shape
    _, N = dy.shape
    
    dw = torch.empty((G, K, N), device=x.device, dtype=torch.float32)

    BK = 64
    BN = 64
    BM = 32

    grid = (
        G,
        triton.cdiv(K, BK),
        triton.cdiv(N, BN),
    )

    MOE_grad[grid](
        x, dy, dw, group_offsets,
        G=G, M_total=M_total, K=K, N=N,
        stride_xm=x.stride(0), stride_xk=x.stride(1),
        stride_dym=dy.stride(0), stride_dyn=dy.stride(1),
        stride_dwg=dw.stride(0), stride_dwk=dw.stride(1), stride_dwn=dw.stride(2),
        BM=BM, BN=BN, BK=BK,
    )
    return dw

def main():
    torch.manual_seed(0)
    device = torch.device("cuda")

    
    G, M_total, K, N = 3, 37, 70, 67
    group_sizes = [11, 13, M_total - 24]
    group_offsets = torch.tensor(
        [0, *torch.tensor(group_sizes).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    x = torch.randn((M_total, K), device=device, dtype=torch.float16)
    dy = torch.randn((M_total, N), device=device, dtype=torch.float16)

    dw = moe_expert_weight_grad(x, dy, group_offsets, G)

    # Reference: each expert only accumulates over its token slice.
    expected = torch.empty((G, K, N), device=device, dtype=torch.float32)
    for g in range(G):
        start = int(group_offsets[g].item())
        end = int(group_offsets[g + 1].item())
        expected[g] = x[start:end].float().transpose(0, 1) @ dy[start:end].float()

    torch.testing.assert_close(dw, expected, rtol=1e-2, atol=1e-2)
    print("MOE expert weight gradient: correctness check passed")


if __name__ == "__main__":
    main()
    