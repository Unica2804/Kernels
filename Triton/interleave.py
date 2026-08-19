import triton
import triton.language as tl
import torch

@triton.jit
def interleave_kernel(A_ptr, B_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    read_offsets = pid*BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = read_offsets < N
    a = tl.load(A_ptr+read_offsets,mask=mask)
    b = tl.load(B_ptr+read_offsets,mask=mask)

    out_start = pid*(2*BLOCK_SIZE)
    even_offset =  2*tl.arange(0,BLOCK_SIZE)
    odd_offset = even_offset+1

    tl.store(output_ptr+out_start+even_offset,a,mask=mask)
    tl.store(output_ptr+out_start+odd_offset,b,mask=mask)


def interleave(A: torch.Tensor, B: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256

    grid = (triton.cdiv(N, BLOCK_SIZE),)

    interleave_kernel[grid](A, B, output, N, BLOCK_SIZE=BLOCK_SIZE)

def main():
    N = 1024
    A = torch.arange(N, device='cuda', dtype=torch.float32)
    B = torch.arange(N, device='cuda', dtype=torch.float32) + N
    output = torch.empty(2*N, device='cuda', dtype=torch.float32)

    interleave(A, B, output, N)

    print("A:", A)
    print("B:", B)
    print("Output:", output)

if __name__ == "__main__":
    main()