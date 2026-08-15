import torch
from torch.utils.cpp_extension import load


spmv_module = load(
    name="spmv_module",
    sources=["CUDA/src/spMV.cu"],
    extra_cuda_cflags=['-ccbin=/usr/bin/g++-13'],
    verbose=True
)


M, N = 1000, 10000
sparsity = 0.95  


A_dense = torch.randn(M, N, device="cuda", dtype=torch.float32)


mask = torch.rand(M, N, device="cuda") < (1 - sparsity)
A = A_dense * mask.float()

v = torch.randn(N, device="cuda", dtype=torch.float32)

y = spmv_module.spmv_csr(A, v)
print("Result of SpMV (y):", y.shape)