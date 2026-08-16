import torch
import triton
import triton.language as tl
import triton.testing
import torch

autotune_configs = [
    triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=4),
]

@triton.autotune(
    configs=autotune_configs,
    key=['input_size', 'kernel_size'],
)

# @triton.jit
# def conv1d_kernel(input_ptr, kernel_ptr, output_ptr, input_size, kernel_size, BLOCK_SIZE: tl.constexpr):
#     pid = tl.program_id(0)
#     offset_base = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
#     c = tl.zeros((BLOCK_SIZE,), dtype=input_ptr.type.element_ty)
#     for j in tl.range(kernel_size):
#         input = tl.load(input_ptr + offset_base + j, mask=(offset_base + j < input_size), other=0.0)
#         kernel_j = tl.load(kernel_ptr + j)
#         c += input * kernel_j
#     tl.store(output_ptr + offset_base, c, mask=offset_base < input_size - kernel_size + 1)

@triton.jit
def conv1d_kernel(input, kernel, output, input_size, kernel_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    out_size = input_size - kernel_size +1
    a_ptr = tl.make_block_ptr(
        base = input,
        shape = (input_size,),
        strides = (1,),
        offsets = (pid*BLOCK_SIZE,),
        block_shape = (BLOCK_SIZE,),
        order = (0,),
    )

    y_ptr = tl.make_block_ptr(
        base = output,
        shape = (out_size,),
        strides = (1,),
        offsets = (pid*BLOCK_SIZE,),
        block_shape = (BLOCK_SIZE,),
        order = (0,),
    )
    acc = tl.zeros((BLOCK_SIZE,),dtype = tl.float32)

    for k in range(kernel_size):
        w_k = tl.load(kernel+k)
        a_tile = tl.load(a_ptr, boundary_check=(0,), padding_option = 'zero')
        acc += w_k * a_tile
        a_ptr = a_ptr.advance((1,))
    acc = acc.cast(tl.bfloat16)
    tl.store(y_ptr, acc, boundary_check = (0,))

def conv1d(input:torch.Tensor, kernel:torch.Tensor) -> torch.Tensor:
    assert input.dim() == 1 and kernel.dim() == 1, "Input and kernel must be 1D tensors"
    input_size = input.size(0)
    kernel_size = kernel.size(0)
    output = torch.empty(input_size - kernel_size + 1, device=input.device, dtype=input.dtype)
    grid = lambda meta: (triton.cdiv(input_size - kernel_size + 1, meta['BLOCK_SIZE']),)
    conv1d_kernel[grid](input, kernel, output, input_size, kernel_size)
    return output

def benchmark_conv1d(input_size=1024, kernel_size=24):
    device = "cuda"
    input_tensor = torch.randn(input_size, device=device, dtype=torch.bfloat16)
    kernel_tensor = torch.randn(kernel_size, device=device, dtype=torch.bfloat16)

    # torch_c = torch.nn.functional.conv1d(input_tensor.view(1, 1, -1), kernel_tensor.view(1, 1, -1)).view(-1)
    # triton_c = conv1d(input_tensor, kernel_tensor)

    # torch.testing.assert_close(torch_c, triton_c, rtol=5e-2, atol=5e-2)

    triton_ms = triton.testing.do_bench(lambda: conv1d(input_tensor, kernel_tensor))
    torch_ms = triton.testing.do_bench(lambda: torch.nn.functional.conv1d(input_tensor.view(1, 1, -1), kernel_tensor.view(1, 1, -1)).view(-1))

    flops = 2 * input_size * kernel_size
    triton_tflops = (flops / (triton_ms * 1e-3)) / 1e12 
    torch_tflops = (flops / (torch_ms * 1e-3)) / 1e12 

    # Print Formatted Table
    print(f"Benchmark Results (Input size: {input_size}, Kernel size: {kernel_size}):")
    print(f"{'Implementation':<24} | {'Time (ms)':<10} | {'Throughput (TFLOPS)':<18}")
    print("-" * 60)
    print(f"{'Triton':<24} | {triton_ms:<10.3f} | {triton_tflops:<18.2f}")
    print(f"{'PyTorch':<24} | {torch_ms:<10.3f} | {torch_tflops:<18.2f}")
    print(f"Best config: {conv1d_kernel.best_config}\n")

if __name__ == "__main__":
    benchmark_conv1d(input_size=2048, kernel_size=256)
    benchmark_conv1d(input_size=2048, kernel_size=128)
    benchmark_conv1d(input_size=4096, kernel_size=128)
    benchmark_conv1d(input_size=4096, kernel_size=256)