import torch
import triton
import triton.language as tl

@triton.jit
def conv_kernel(
    input_ptr,
    kernel_ptr,
    output_ptr,
    i_m,i_n,
    k_m,k_n,
    o_m,o_n,
    stride_im,
    stride_in,
    stride_km,
    stride_kn,
    stride_om,
    stride_on,
    BLOCK_M:tl.constexpr,
    BLOCK_N:tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0,BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0,BLOCK_N)

    mask_out = (offs_m[:,None] < o_m) & (offs_n[None,:] < o_n)

    acc = tl.zeros((BLOCK_M,BLOCK_N),dtype = tl.float32)

    for i in range(k_m):
        for j in range(k_n):
            k_ptrs = kernel_ptr+i*stride_km+j*stride_kn
            weight = tl.load(k_ptrs)

            in_m = offs_m+i
            in_n = offs_n+j

            in_mask = (in_m[:,None]<i_m) & (in_n[None,:]<i_n)
            in_ptrs = input_ptr + (in_m[:,None]*stride_im+in_n[None,:]*stride_in)
            inp = tl.load(in_ptrs,mask = in_mask, other = 0.0)

            acc+= weight*inp
    out_ptrs = output_ptr + (offs_m[:,None]*stride_om+offs_n[None,:]*stride_on)
    tl.store(out_ptrs,acc,mask=mask_out)        


@triton.jit
def conv_kernel_blk(

    input_ptr,
    kernel_ptr,
    output_ptr,
    i_m,i_n,
    k_m,k_n,
    o_m,o_n,
    stride_im,
    stride_in,
    stride_km,
    stride_kn,
    stride_om,
    stride_on,
    BLOCK_M:tl.constexpr,
    BLOCK_N:tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M 
    offs_n = pid_n * BLOCK_N

    input_blk = tl.make_block_ptr(
        base = input_ptr,
        shape = (i_m,i_n),
        strides = (stride_im,stride_in),
        offsets = (offs_m,offs_n),
        block_shape = (BLOCK_M,BLOCK_N),
        order = (0,1)
    )

    output_blk = tl.make_block_ptr(
        base = output_ptr,
        shape = (o_m,o_n),
        strides = (stride_om,stride_on),
        offsets = (offs_m,offs_n),
        block_shape = (BLOCK_M,BLOCK_N),
        order = (0,1)
    )

    acc = tl.zeros((BLOCK_M,BLOCK_N),dtype = tl.float32)

    for i in range(k_m):
        for j in range(k_n):
            k_ptrs = kernel_ptr+i*stride_km+j*stride_kn
            weight = tl.load(k_ptrs)
            ip_bl = tl.advance(input_blk, (i,j))
            input = tl.load(ip_bl,boundary_check=(0,1),padding_option = "zero")
            acc+= weight*input
    tl.store(output_blk,acc,boundary_check=(0,1))

            


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
):
    o_m = input_rows - kernel_rows + 1
    o_n = input_cols - kernel_cols + 1

    BLOCK_M = 16
    BLOCK_N = 16

    # Determine strides safely whether tensors are 1D or 2D
    stride_im = input.stride(0) if input.ndim == 2 else input_cols
    stride_in = input.stride(1) if input.ndim == 2 else 1

    stride_km = kernel.stride(0) if kernel.ndim == 2 else kernel_cols
    stride_kn = kernel.stride(1) if kernel.ndim == 2 else 1

    stride_om = output.stride(0) if output.ndim == 2 else o_n
    stride_on = output.stride(1) if output.ndim == 2 else 1

    grid = (triton.cdiv(o_m, BLOCK_M), triton.cdiv(o_n, BLOCK_N))

    # conv_kernel[grid](
    #     input,
    #     kernel,
    #     output,
    #     input_rows, input_cols,
    #     kernel_rows, kernel_cols,
    #     o_m, o_n,
    #     stride_im, stride_in,
    #     stride_km, stride_kn,
    #     stride_om, stride_on,
    #     BLOCK_M=BLOCK_M,
    #     BLOCK_N=BLOCK_N,
    # )

    conv_kernel_blk[grid](
        input,
        kernel,
        output,
        input_rows, input_cols,
        kernel_rows, kernel_cols,
        o_m, o_n,
        stride_im, stride_in,
        stride_km, stride_kn,
        stride_om, stride_on,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

def main():
    #Parameters
    input = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
    kernel = torch.randn(3, 3, device='cuda', dtype=torch.float32)
    output = torch.empty(1022, 1022, device='cuda', dtype=torch.float32)
    triton_ms = triton.testing.do_bench(lambda: solve(input, kernel, output, 1024, 1024, 3, 3))
    torch_ms = triton.testing.do_bench(lambda: torch.nn.functional.conv2d(input.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=0))
    print(f"Torch 2D convolution kernel time: {torch_ms:.2f} ms,dtype: {input.dtype}, shape: {input.shape}")
    print(f"Triton 2D convolution kernel time: {triton_ms:.2f} ms,dtype: {input.dtype}, shape: {input.shape}")

if __name__ == "__main__":
    main()
