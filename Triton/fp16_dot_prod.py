import torch
import triton
import triton.language as tl

@triton.jit
def mul_1D(
    a,b,c,N,
    BN:tl.constexpr,
):
    pid=tl.program_id(0)
    offset = pid*BN + tl.arange(0,BN)
    mask = offset < N
    a_ptr = a+offset
    b_ptr = b+offset
    a_val = tl.load(a_ptr,mask=mask,other=0.0)
    b_val = tl.load(b_ptr,mask=mask,other=0.0)

    prod = a_val.to(tl.float32) * b_val.to(tl.float32)

    acc = tl.sum(prod,axis=0)

    tl.atomic_add(c,acc)

# A, B, result are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    GRID = (triton.cdiv(N,BLOCK_SIZE),)
    mul_1D[GRID](
        A,B,result,N,BLOCK_SIZE
    )

def main():
    N = 1000000
    A = torch.rand(N,device='cuda',dtype=torch.float16)
    B = torch.rand(N,device='cuda',dtype=torch.float16)
    result = torch.zeros(1,device='cuda',dtype=torch.float32)

    solve(A,B,result,N)
    torch_result = torch.dot(A.to(torch.float32),B.to(torch.float32))

    print("Result:",result.item())
    print("Torch Result:",torch_result.item())
    print("Difference:",abs(result.item() - torch_result.item()))
if __name__ == "__main__":
    main()