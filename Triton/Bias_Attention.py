import torch
import triton
import triton.language as tl

@triton.jit
def attn_bias(
    Q,K,V,output,
    M,N,d,
    stride_qm,stride_qd,
    stride_kn,stride_kd,
    stride_vn,stride_vd,
    stride_om,stride_od,
    alpha,
    BM:tl.constexpr,
    BN:tl.constexpr,
    Bd:tl.constexpr,
):

    pid = tl.program_id(0)

    offs_m = pid*BM+tl.arange(0,BM)
    offs_n = tl.arange(0,BN)
    offs_d = tl.arange(0,Bd)

    q_mask = (offs_m[:,None] < M) & (offs_d[None,:]<d)
    q_ptr = Q + offs_m[:,None] * stride_qm + offs_d[None,:] * stride_qd
    q_val = tl.load(q_ptr,mask=q_mask,other=0.0)

    scale = 1.0/tl.sqrt(d.to(tl.float32))
    m_prev = tl.full([BM],float("-inf"),dtype=tl.float32)
    l_prev = tl.zeros([BM],dtype=tl.float32)
    acc = tl.zeros([BM,Bd],dtype=tl.float32)

    for start_n in range(0,N,BN):

        curr_n = start_n + offs_n

        k_mask = (curr_n[:,None]<N) & (offs_d[None,:]<d)
        k_ptr = K + curr_n[:,None] * stride_kn + offs_d[None,:] * stride_kd
        k_val = tl.load(k_ptr,mask=k_mask,other=0.0)

        v_mask = (curr_n[:,None]<N) & (offs_d[None,:]<d)
        v_ptr = V + curr_n[:,None] * stride_vn + offs_d[None,:] * stride_vd
        v_val = tl.load(v_ptr,mask=v_mask,other=0.0)

        score = tl.dot(q_val,tl.trans(k_val),input_precision="ieee") * scale

        delta = offs_m[:,None] - curr_n[None,:]
        score = score + alpha * delta.to(tl.float32)

        valid_mask = (offs_m[:,None]<M) & (curr_n[None,:]<N)
        score = tl.where(valid_mask,score,float("-inf"))

        m_curr = tl.max(score,axis=1)
        m_new = tl.maximum(m_prev,m_curr)

        corr=tl.exp(m_prev-m_new)
        p = tl.exp(score-m_new[:,None])

        l_new = l_prev * corr + tl.sum(p,axis=1)

        acc = acc*corr[:,None]
        acc=tl.dot(p.to(v_val.dtype),v_val,acc,input_precision="ieee")

        m_prev = m_new
        l_prev = l_new

    acc = acc / l_prev[:,None]
    out_mask = (offs_m[:, None] < M) & (offs_d[None, :] < d)
    out_ptrs = output + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=out_mask)

# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    N: int,
    d: int,
    alpha: float,
):
    BM=32
    BN=32
    Bd=max(16,triton.next_power_of_2(d))
    grid = (triton.cdiv(M,BM),)
    attn_bias[grid](
        Q, K, V, output,
        M, N, d,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        output.stride(0), output.stride(1),
        alpha,
        BM=BM,
        BN=BN,
        Bd=Bd,
    )
    return output

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Bias attention benchmark requires a CUDA device")

    M, N, d = 1024, 1024, 64 
    alpha = 0.01
    dtype = torch.bfloat16

    Q = torch.randn((M, d), device="cuda", dtype=dtype)
    K = torch.randn((N, d), device="cuda", dtype=dtype)
    V = torch.randn((N, d), device="cuda", dtype=dtype)
    output = torch.empty((M, d), device="cuda", dtype=dtype)

    triton_output = solve(Q, K, V, output, M, N, d, alpha)
    rows = torch.arange(M, device="cuda", dtype=torch.float32)[:, None]
    cols = torch.arange(N, device="cuda", dtype=torch.float32)[None, :]
    bias = alpha * (rows - cols)
    scores = (Q.float() @ K.float().transpose(0, 1)) / (d ** 0.5) + bias
    torch_output = torch.softmax(scores, dim=-1) @ V.float()
    torch.testing.assert_close(
        triton_output.float(), torch_output, rtol=2e-2, atol=2e-2
    )
    print("Correctness check: passed")

    benchmark_ms = triton.testing.do_bench(
        lambda: solve(Q, K, V, output, M, N, d, alpha)
    )
    bytes_moved = (
        Q.numel() + K.numel() + V.numel() + output.numel()
    ) * Q.element_size()
    bandwidth_gb_s = bytes_moved / (benchmark_ms * 1e6)
    print(f"Triton time: {benchmark_ms:.2f} ms")
    print(f"Memory bandwidth: {bandwidth_gb_s:.2f} GB/s")

if __name__ == "__main__":
    main()