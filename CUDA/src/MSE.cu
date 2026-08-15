// [ Input Predictions (y_hat) & Targets (y) ]
//                    │
//                    ▼  (Grid-Strided Loop + float4 Vectorized Loads)
//     ┌──────────────┴──────────────┐
//     │ Subtraction & Squaring      │  --> (y_hat - y)^2 per element
//     └──────────────┬──────────────┘
//                    │
//                    ▼  (Warp-level Shuffles: __shfl_down_sync)
//     ┌──────────────┴──────────────┐
//     │ Intra-Warp Sum Reduction    │
//     └──────────────┬──────────────┘
//                    │
//                    ▼  (Shared Memory across warps)
//     ┌──────────────┴──────────────┐
//     │ Intra-Block Sum Reduction   │
//     └──────────────┬──────────────┘
//                    │
//                    ▼  (Thread 0 of each block)
//     ┌──────────────┴──────────────┐
//     │ Atomic Add to Global Loss   │  --> atomicAdd(loss, block_sum / N)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>

__device__ inline float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline float blockReduceSum(float val) {
    static __shared__ float shared_sum[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    val = warpReduceSum(val); 

    if (lane == 0) {
        shared_sum[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared_sum[lane] : 0.0f;
    if (warp_id == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

__global__ void mse_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ loss,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;

    for (int i = idx; i < N; i += stride) {
        float diff = predictions[i] - targets[i];
        sum += diff * diff;
    }

    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) {
        atomicAdd(loss, sum / static_cast<float>(N));
    } 

}

torch::Tensor mse_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto loss = torch::zeros({1}, predictions.options());
    int N = predictions.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    mse_kernel<<<blocks, threads>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), loss.data_ptr<float>(), N);
    cudaDeviceSynchronize();
    return loss;
}

PYBIND11_MODULE(mse_kernel_cuda, m) {
    m.def("calc", &mse_cuda, "Mean Squared Error kernel");
}