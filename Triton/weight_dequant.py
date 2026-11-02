import torch
import triton
import triton.language as tl

@triton.jit
def dequant(
    x,s,y,
    M,N,
    stride_xm,stride_xn,
    stride_sm,stride_sn,
    stride_ym,stride_yn,
    tile:tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    scale = tl.load(s + pid_m * stride_sm + pid_n * stride_sn)
    offs_m = pid_m*tile + tl.arange(0,tile)
    offs_n = pid_n*tile + tl.arange(0,tile)

    x_ptr = x + (offs_m[:,None]*stride_xm + offs_n[None,:]*stride_xn)
    y_ptr = y + (offs_m[:,None]*stride_ym + offs_n[None,:]*stride_yn)

    mask = (offs_m[:,None]<M) & (offs_n[None,:]<N)
    x_val = tl.load(x_ptr,mask=mask,other=0.0)
    y_val = x_val.to(scale.dtype) * scale

    tl.store(y_ptr,y_val,mask=mask)


# X, S, Y are tensors on the GPU
def solve(X: torch.Tensor, S: torch.Tensor, Y: torch.Tensor):
    M,N = X.shape
    Tile = 32
    grid = (triton.cdiv(M,Tile),triton.cdiv(N,Tile))
    dequant[grid](
        X,S,Y,
        M,N,
        X.stride(0),X.stride(1),
        S.stride(0),S.stride(1),
        Y.stride(0),Y.stride(1),
        Tile
    )

def main():
    M,N = 1000,1000
    Tile = 32
    X = torch.randint(-128,127,(M,N),device='cuda',dtype=torch.int8)
    S = torch.rand(
        (triton.cdiv(M, Tile), triton.cdiv(N, Tile)),
        device='cuda',
        dtype=torch.float32,
    )
    Y = torch.zeros((M,N),device='cuda',dtype=torch.float32)

    solve(X,S,Y)

    # Verify the result
    expanded_S = S.repeat_interleave(Tile, dim=0).repeat_interleave(Tile, dim=1)
    expected_Y = X.to(torch.float32) * expanded_S[:M, :N]
    assert torch.allclose(Y, expected_Y, atol=1e-5), "Dequantization failed!"
    print("Dequantization successful!")

if __name__ == "__main__":
    main()