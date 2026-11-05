import torch
import triton
import triton.language as tl

@triton.jit
def batch_norm(
    inp_ptr,output_ptr,
    gamma,beta,
    N,C,eps,
    stride_inp_n,stride_inp_c,
    stride_out_n,stride_out_c,
    stride_gamma,stride_beta,
    BN:tl.constexpr,
    BC:tl.constexpr,
):  
    pid = tl.program_id(axis=0)

    offsets_c = pid * BC + tl.arange(0, BC)
    mask_c = offsets_c < C

    sum_x = tl.zeros([BC], dtype=tl.float32)
    for start_n in range(0, N, BN):
        offsets_n = start_n + tl.arange(0, BN)
        mask_n = offsets_n < N
        mask = mask_n[:, None] & mask_c[None, :]
        x_ptrs = inp_ptr + offsets_n[:, None] * stride_inp_n + offsets_c[None, :] * stride_inp_c
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
    mean = sum_x / N

    sum_var = tl.zeros([BC], dtype=tl.float32)
    for start_n in range(0, N, BN):
        offsets_n = start_n + tl.arange(0, BN)
        mask_n = offsets_n < N
        mask = mask_n[:, None] & mask_c[None, :]
        x_ptrs = inp_ptr + offsets_n[:, None] * stride_inp_n + offsets_c[None, :] * stride_inp_c
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean[None, :], 0.0)
        sum_var += tl.sum(diff * diff, axis=0)
    var = sum_var / N
    inv_std = 1/tl.sqrt(var + eps)

    gamma = tl.load(gamma + offsets_c * stride_gamma, mask=mask_c, other=0.0).to(tl.float32)
    beta = tl.load(beta + offsets_c * stride_beta, mask=mask_c, other=0.0).to(tl.float32)

    scale = gamma * inv_std
    shift = beta - mean * scale

    for start_n in range(0, N, BN):
        offsets_n = start_n + tl.arange(0, BN)
        mask_n = offsets_n < N
        mask = mask_n[:, None] & mask_c[None, :]
        x_ptrs = inp_ptr + offsets_n[:, None] * stride_inp_n + offsets_c[None, :] * stride_inp_c
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        y = x*scale[None, :] + shift[None, :]

        out_ptrs = output_ptr + offsets_n[:, None] * stride_out_n + offsets_c[None, :] * stride_out_c
        tl.store(out_ptrs, y, mask=mask)

def solve(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float
):
    N, C = input.shape
    BN = 64
    BC = min(64,triton.next_power_of_2(C))
    output = torch.empty_like(input)

    grid = (triton.cdiv(C, BC),)

    batch_norm[grid](
        input, output,
        gamma, beta,
        N, C, eps,
        input.stride(0), input.stride(1),
        output.stride(0), output.stride(1),
        gamma.stride(0), beta.stride(0),
        BN, BC
    )
    return output

def main():
    torch.manual_seed(42)
    N, C = 1024, 512
    input = torch.randn(N, C, device='cuda', dtype=torch.float32)
    gamma = torch.randn(C, device='cuda', dtype=torch.float32)
    beta = torch.randn(C, device='cuda', dtype=torch.float32)
    eps = 1e-5

    output = solve(input, gamma, beta, eps)

    ref_output = torch.nn.functional.batch_norm(
        input,
        running_mean=None,
        running_var=None,
        weight=gamma,
        bias=beta,
        training=True,
        eps=eps
    )

    torch.testing.assert_close(output, ref_output, atol=1e-5, rtol=1e-5)
    print("Verification passed! Output matches PyTorch reference.")


if __name__ == "__main__":
    main()
