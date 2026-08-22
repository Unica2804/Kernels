# Reference implementation of attention in PyTorch and Triton from https://www.isztld.com/posts/online-softmax.html

import torch
import triton
import triton.language as tl
import math

def torch_attention(q, k, v):
    B,H,N,D = q.shape

    scale = 1.0 / math.sqrt(D)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    attn = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output

@triton.jit
def naive_attention_kernel(
    q_ptr,k_ptr,v_ptr,output_ptr,
    stride_qb,stride_qh,stride_qn,stride_qd,
    stride_kb,stride_kh,stride_kn,stride_kd,
    stride_vb,stride_vh,stride_vn,stride_vd,
    stride_ob,stride_oh,stride_on,stride_od,
    N,D,scale,
    BLOCK_D: tl.constexpr,
):
    # q: [B,H,N,D]
    # k: [B,H,N,D]
    # v: [B,H,N,D]
    # output: [B,H,N,D]
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    q_offset = pid_b * stride_qb + pid_h * stride_qh + pid_n * stride_qn

    d_range = tl.arange(0, BLOCK_D)
    q = tl.load(q_ptr + q_offset + d_range * stride_qd, mask=d_range < D, other=0.0)

    global_max = float("-inf")
    global_sum = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for j in range(N):
        #load K
        k_offset = pid_b * stride_kb + pid_h * stride_kh + j * stride_kn
        k = tl.load(k_ptr + k_offset + d_range * stride_kd, mask=d_range < D, other=0.0)
        
        #compute qk^T
        score = tl.sum(q * k, axis=0) * scale
        new_max = tl.maximum(score, global_max)
        alpha = tl.exp(global_max - new_max)
        global_sum = global_sum * alpha + tl.exp(score - new_max)
        acc = acc * alpha

        # load V and accumulate
        v_offset = pid_b * stride_vb + pid_h * stride_vh + j * stride_vn
        v = tl.load(v_ptr + v_offset + d_range * stride_vd, mask=d_range < D, other=0.0)
        acc += tl.exp(score - new_max) * v
    acc /= global_sum
    output_offset = pid_b * stride_ob + pid_h * stride_oh + pid_n * stride_on
    tl.store(output_ptr + output_offset + d_range * stride_od, acc, mask=d_range < D)

@triton.jit
def online_attention_kernel(
    q_ptr,k_ptr,v_ptr,output_ptr,
    stride_qb,stride_qh,stride_qn,stride_qd,
    stride_kb,stride_kh,stride_kn,stride_kd,
    stride_vb,stride_vh,stride_vn,stride_vd,
    stride_ob,stride_oh,stride_on,stride_od,
    N,D,scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # q: [B,H,N,D]
    # k: [B,H,N,D]
    # v: [B,H,N,D]
    # output: [B,H,N,D]
    
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    mask_m = offs_m < N
    q = tl.load(q_ptrs, mask = mask_m[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # loop over key/values blocks
    for block_start in range(0, N, BLOCK_N):
        offs_n = block_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        k_ptrs = k_ptr + pid_bh * stride_kh + offs_n[:, None]*stride_kn + offs_d[None, :]*stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        scores = tl.dot(q,tl.trans(k)) * scale
        #mask out invalid positions
        scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, -float('inf'))
        # max of this block
        m_block = tl.max(scores, axis=1)
        # new global max
        m_new = tl.maximum(m_i, m_block)
        # correction factors
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_block - m_new)

        # rescale running sum and accumulator
        l_i = l_i * alpha
        acc = acc * alpha[:, None]

        #compute softmax for this block
        p = tl.exp(scores - m_new[:, None])

        #update running sum
        l_i += tl.sum(p, axis=1)

        #load V
        v_ptrs = v_ptr + pid_bh * stride_vh + offs_n[:, None]*stride_vn + offs_d[None, :]*stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        #accumulate weighted values: [BLOCK_M, D]
        acc += tl.dot(p.to(v.dtype), v)

        #update max
        m_i = m_new
    # Normalize accumulator
    acc /= l_i[:, None]
    # store output
    output_ptrs = output_ptr + pid_bh * stride_oh + offs_m[:, None]*stride_on + offs_d[None, :]*stride_od
    tl.store(output_ptrs, acc, mask=mask_m[:, None])


def naive_attention(q,k,v):
    B,H,N,D = q.shape
    output = torch.empty_like(q)
    grid = (B,H,N)
    scale = 1.0 / math.sqrt(D)
    naive_attention_kernel[grid](
        q,k,v,output,
        q.stride(0),q.stride(1),q.stride(2),q.stride(3),
        k.stride(0),k.stride(1),k.stride(2),k.stride(3),
        v.stride(0),v.stride(1),v.stride(2),v.stride(3),
        output.stride(0),output.stride(1),output.stride(2),output.stride(3),
        N,D,scale,
        BLOCK_D=64,
        num_warps=4,
        num_stages=2,
    )
    return output

def online_attention(q,k,v):
    B,H,N,D = q.shape
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    output = torch.empty_like(q)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(D)
    # Grid: (num_q_blocks, batch*head)
    grid = (triton.cdiv(N, BLOCK_M), B*H)
    scale = 1.0 / math.sqrt(D)
    online_attention_kernel[grid](
        q,k,v,output,
        q.stride(0),q.stride(1),q.stride(2),q.stride(3),
        k.stride(0),k.stride(1),k.stride(2),k.stride(3),
        v.stride(0),v.stride(1),v.stride(2),v.stride(3),
        output.stride(0),output.stride(1),output.stride(2),output.stride(3),
        N,D,scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=8,
        num_stages=3,
    )
    return output

def benchmark(fn, Q, K, V, name, warmup=10, iters=100):
    # Warmup
    for _ in range(warmup):
        _ = fn(Q, K, V)
    
    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        _ = fn(Q, K, V)
    end.record()
    torch.cuda.synchronize()
    
    ms = start.elapsed_time(end) / iters
    print(f"{name}: {ms:.2f} ms , dtype: {Q.dtype}, shape: {Q.shape}")
    return ms

def main():
    B,H,N,D = 4,32,512,64
    Q = torch.randn(B,H,N,D,device='cuda',dtype=torch.bfloat16)
    K = torch.randn(B,H,N,D,device='cuda',dtype=torch.bfloat16)
    V = torch.randn(B,H,N,D,device='cuda',dtype=torch.bfloat16)
    naive_ms = benchmark(naive_attention, Q, K, V, "Naive Attention")
    online_ms = benchmark(online_attention, Q, K, V, "Online Attention")
    torch_ms = benchmark(torch_attention, Q, K, V, "Torch Attention")
    sdpa_torch_ms = benchmark(torch.nn.functional.scaled_dot_product_attention, Q, K, V, "Torch SDPA")

if __name__ == "__main__":
    main()