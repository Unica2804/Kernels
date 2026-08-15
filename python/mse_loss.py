import torch
from torch.utils.cpp_extension import load

mse_kernel = load(
    name = "mse_kernel_cuda",
    sources = ["CUDA/src/MSE.cu"],
    extra_cuda_cflags=['-ccbin=/usr/bin/g++-13'],
    verbose=True
)

preds = torch.randn(32, 10, device="cuda", dtype=torch.float32)
targets = torch.randn(32, 10, device="cuda", dtype=torch.float32)

loss = mse_kernel.calc(preds, targets)
true_loss = torch.nn.functional.mse_loss(preds, targets)
print(torch.allclose(loss, true_loss, atol=1e-5, rtol=1e-4))