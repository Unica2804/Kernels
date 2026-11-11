"""
Grouped Query Attention (GQA) implementation in Triton.
"""

import triton
import triton.language as tl
import torch

@triton.jit
def gqa_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    num_q_heads, num_kv_heads,
    seq_len, head_dim,
    stride_qh, stride_qm, stride_qd,
    stride_kh, stride_kn, stride_kd,
    stride_vh, stride_vn, stride_vd,
    stride_oh, stride_om, stride_od,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
) :
    pid_m = tl.program_id(0)
    pid_q = tl.program_id(1)

    group_size = num_q_heads // num_kv_heads
    pid_kv = pid_q // group_size

    # Base ptrs
    q_base = q_ptr + pid_q * stride_qh
    k_base = k_ptr + pid_kv * stride_kh
    v_base = v_ptr + pid_kv * stride_vh
    out_base = o_ptr + pid_q * stride_oh

    # Offsets
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, BD)

    # Load Query tile [BM, BD]
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

    scale = 1.0 / tl.sqrt(head_dim.to(tl.float32))

    # Init Online Softmax 
    m_prev = tl.full([BM], float('-inf'), dtype=tl.float32)
    l_prev = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros((BM, BD), dtype=tl.float32)

    # Stream through K and V in tiles of size [BN, BD]
    for start_n in range(0, seq_len, BN):
        curr_n = start_n + offs_n

        #load K tile [BN, BD]
        k_ptrs = k_base + curr_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = (curr_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Load V tile [BN, BD]
        v_ptrs = v_base + curr_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = (curr_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        v_tile = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Compute QK^T * scale
        s = tl.dot(q_tile, tl.trans(k_tile)) * scale

        # Mask out-of-bounds positions
        valid_mask = (offs_m[:, None] < seq_len) & (curr_n[None, :] < seq_len)
        s = tl.where(valid_mask, s, float('-inf'))

        # Softmax update
        m_curr = tl.max(s, axis=1)
        m_new = tl.maximum(m_curr, m_prev)

        corr = tl.exp(m_prev - m_new)
        p = tl.exp(s - m_new[:, None])

        l_new = l_prev * corr + tl.sum(p, axis=1)

        acc = acc * corr[:, None]
        acc = tl.dot(p.to(v_tile.dtype), v_tile, acc)

        m_prev = m_new
        l_prev = l_new
    
    # Normalize and write output
    acc = acc / l_prev[:, None]
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    out_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    tl.store(out_ptrs, acc, mask=out_mask)

def gqa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
): 
    num_q_heads, seq_len, head_dim = Q.shape
    num_kv_heads = K.shape[0]

    Out = torch.empty_like(Q)

    if Q.ndim == 3:
        stride_qh, stride_qm, stride_qd = Q.stride(0), Q.stride(1), Q.stride(2)
        stride_kh, stride_kn, stride_kd = K.stride(0), K.stride(1), K.stride(2)
        stride_vh, stride_vn, stride_vd = V.stride(0), V.stride(1), V.stride(2)
        stride_oh, stride_om, stride_od = Out.stride(0), Out.stride(1), Out.stride(2)
    else:
        stride_qh, stride_qm, stride_qd = seq_len * head_dim, head_dim, 1
        stride_kh, stride_kn, stride_kd = seq_len * head_dim, head_dim, 1
        stride_vh, stride_vn, stride_vd = seq_len * head_dim, head_dim, 1
        stride_oh, stride_om, stride_od = seq_len * head_dim, head_dim, 1
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = triton.next_power_of_2(head_dim)

    grid = (triton.cdiv(seq_len, BLOCK_M), num_q_heads)

    gqa_kernel[grid](
        Q, K, V, Out,
        num_q_heads, num_kv_heads,
        seq_len, head_dim,
        stride_qh, stride_qm, stride_qd,
        stride_kh, stride_kn, stride_kd,
        stride_vh, stride_vn, stride_vd,
        stride_oh, stride_om, stride_od,
        BM=BLOCK_M,
        BN=BLOCK_N,
        BD=BLOCK_D,
    )
    return Out

def main():
    num_q_heads = 8
    num_kv_heads = 2
    seq_len = 128
    head_dim = 64

    Q = torch.randn((num_q_heads, seq_len, head_dim), device='cuda', dtype=torch.float16)
    K = torch.randn((num_kv_heads, seq_len, head_dim), device='cuda', dtype=torch.float16)
    V = torch.randn((num_kv_heads, seq_len, head_dim), device='cuda', dtype=torch.float16)

    Out = gqa(Q, K, V)
    print(Out.shape)

    torch_out = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False,enable_gqa=True)
    print(torch_out.shape)

    torch.testing.assert_close(Out, torch_out, rtol=1e-2, atol=1e-2)
    print("correctness passed!")

if __name__ == "__main__":
    main()