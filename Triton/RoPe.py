import triton
import triton.language as tl
import torch

@triton.jit
def rope(
    q,cos,sin,out,
    M,D,
    stride_qm, stride_qd, stride_cm, stride_cd, stride_om, stride_od,
    BM:tl.constexpr,
    HALF_BD:tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Compute offsets for the M dimension
    offs_m = pid * BM + tl.arange(0, BM)
    mask_m = offs_m < M
    
    # Compute offsets for the D dimension
    half_D = D // 2
    offs_d_left = tl.arange(0, HALF_BD)
    offs_d_right = half_D + tl.arange(0, HALF_BD)
    mask_d = offs_d_left < half_D
    
    # Create a 2D mask for the (M, D/2) grid
    mask = mask_m[:, None] & mask_d[None, :]

    # Compute pointers
    q_ptr_left = q + offs_m[:, None] * stride_qm + offs_d_left[None, :] * stride_qd
    q_ptr_right = q + offs_m[:, None] * stride_qm + offs_d_right[None, :] * stride_qd
    cos_ptr = cos + offs_m[:, None] * stride_cm + offs_d_left[None, :] * stride_cd
    sin_ptr = sin + offs_m[:, None] * stride_cm + offs_d_left[None, :] * stride_cd
    out_ptr_left = out + offs_m[:, None] * stride_om + offs_d_left[None, :] * stride_od
    out_ptr_right = out + offs_m[:, None] * stride_om + offs_d_right[None, :] * stride_od

    # Load inputs
    q_left = tl.load(q_ptr_left, mask=mask, other=0.0)
    q_right = tl.load(q_ptr_right, mask=mask, other=0.0)
    cos_val = tl.load(cos_ptr, mask=mask, other=0.0)
    sin_val = tl.load(sin_ptr, mask=mask, other=0.0)

    #compute rope
    out_left = q_left * cos_val - q_right * sin_val
    out_right = q_left * sin_val + q_right * cos_val

    # store results
    tl.store(out_ptr_left, out_left, mask=mask)
    tl.store(out_ptr_right, out_right, mask=mask)

def rope_kernel(q,cos,sin,out):
    M,D = q.shape
    assert D % 2 == 0, "D must be even"
    BM = 128
    HALF_BD = triton.next_power_of_2(D // 2)

    # Compute strides
    stride_qm, stride_qd = q.stride()
    stride_cm, stride_cd = cos.stride()
    stride_om, stride_od = out.stride()

    grid = (triton.cdiv(M, BM),)
    
    rope[grid](
        q, cos, sin, out,
        M, D,
        stride_qm, stride_qd, stride_cm, stride_cd, stride_om, stride_od,
        BM=BM,
        HALF_BD=HALF_BD,
    )

def rope_torch(q, cos, sin):
    """Reference PyTorch implementation matching the Triton kernel."""
    half_D = q.shape[-1] // 2
    q_left = q[..., :half_D]
    q_right = q[..., half_D:]
    out_left = q_left * cos - q_right * sin
    out_right = q_left * sin + q_right * cos
    return torch.cat((out_left, out_right), dim=-1)

def main():
    M, D = 1024, 512
    q = torch.randn(M, D, device='cuda', dtype=torch.float32)
    cos = torch.randn(M, D // 2, device='cuda', dtype=torch.float32)
    sin = torch.randn(M, D // 2, device='cuda', dtype=torch.float32)
    out = torch.empty_like(q)

    rope_kernel(q, cos, sin, out)
    expected = rope_torch(q, cos, sin)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
    print("RoPE kernel matches PyTorch reference")

    

if __name__ == "__main__":
    main()