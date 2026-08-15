import torch
from torch.utils.cpp_extension import load

# JIT-compile the CUDA source file on the fly
fused_cross_entropy_module = load(
    name="fused_cross_entropy_module",
    sources=["CUDA/src/fused_cross_entropy.cu"],
    # allow unsupported host compiler (nvcc check) as a temporary workaround
    extra_cuda_cflags=['-ccbin=/usr/bin/g++-13'],
    verbose=True
)

logits = torch.randn(32, 10, device="cuda", dtype=torch.float32)
labels = torch.randint(0, 10, (32,), device="cuda", dtype=torch.int64)

loss = fused_cross_entropy_module.calc(logits, labels)
true_loss = torch.nn.functional.cross_entropy(logits, labels)
print("calcLoss:", loss.item())
print("\ntrueLoss:", true_loss.item())