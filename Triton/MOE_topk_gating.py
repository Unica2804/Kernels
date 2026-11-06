import triton
import triton.language as tl
import torch

@triton.jit
def moe_topk_gating(
    logits_ptr,
    topkidx_ptr,
    topkwt_ptr,
    M, E, K,
    stride_lm, stride_le,
    stride_im, stride_ik,
    stride_wm, stride_wk,
    BE:tl.constexpr, 
    BK:tl.constexpr,
):
    row_id = tl.program_id(axis=0)

    cols = tl.arange(0, BE)
    cols_mask = cols < E

    curr_ptr = logits_ptr + row_id * stride_lm + cols * stride_le
    curr_logits = tl.load(curr_ptr, mask=cols_mask, other=float('-inf'))

    k_range = tl.arange(0, BK)
    topk_vals = tl.full([BK], float('-inf'), dtype=tl.float32)
    topk_idx = tl.full([BK], 0, dtype=tl.int32)

    for j in range(K):
        max_val = tl.max(curr_logits, axis=0)
        
        is_max = curr_logits == max_val
        cand_idx = tl.where(is_max, cols, BE)
        max_idx = tl.min(cand_idx, axis=0)

        slot_mask = k_range == j
        topk_vals = tl.where(slot_mask, max_val, topk_vals)
        topk_idx = tl.where(slot_mask, max_idx, topk_idx)

        curr_logits = tl.where(cols == max_idx, float('-inf'), curr_logits)
    
    k_mask = k_range < K
    v_max = tl.max(tl.where(k_mask, topk_vals, float('-inf')), axis=0)
    exp_vals = tl.where(k_mask, tl.exp(topk_vals - v_max), 0.0)
    denom = tl.sum(exp_vals, axis=0)
    weights = exp_vals / denom

    out_idx_ptr = topkidx_ptr + row_id * stride_im + k_range * stride_ik
    out_wt_ptr = topkwt_ptr + row_id * stride_wm + k_range * stride_wk
    tl.store(out_idx_ptr, topk_idx, mask=k_mask)
    tl.store(out_wt_ptr, weights, mask=k_mask)

def moe_topk_gating_kernel(logits:torch.Tensor, K:int):
    M, E = logits.shape

    topkidx = torch.empty((M, K), dtype=torch.int32, device=logits.device)
    topkwt = torch.empty((M, K), dtype=torch.float32, device=logits.device)

    BE = triton.next_power_of_2(E)
    BK = triton.next_power_of_2(K)

    grid = (M,)

    moe_topk_gating[grid](
        logits_ptr=logits,
        topkidx_ptr=topkidx,
        topkwt_ptr=topkwt,
        M=M, E=E, K=K,
        stride_lm=logits.stride(0), stride_le=logits.stride(1),
        stride_im=topkidx.stride(0), stride_ik=topkidx.stride(1),
        stride_wm=topkwt.stride(0), stride_wk=topkwt.stride(1),
        BE=BE, BK=BK
    )

    return topkidx, topkwt

def main():
    device = "cuda"
    M = 4096
    E = 64
    k = 4

    torch.manual_seed(0)
    logits = torch.randn((M, E), device=device, dtype=torch.float32)

    # Triton Run
    triton_idx, triton_w = moe_topk_gating_kernel(logits, k)

    # PyTorch Reference
    torch_vals, torch_idx = torch.topk(logits, k=k, dim=-1, largest=True, sorted=True)
    torch_w = torch.softmax(torch_vals, dim=-1)

    # Assertions
    torch.testing.assert_close(triton_idx, torch_idx.to(torch.int32))
    torch.testing.assert_close(triton_w, torch_w, atol=1e-5, rtol=1e-5)
    print("Verification Passed: 100% exact match")

if __name__ == "__main__":
    main()