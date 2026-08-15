/// log(sum from c=1 to C of exp(x[i,c])) = m_i + log(sum from c=1 to C of exp(x[i,c] - m_i))
/// L_i = log( sum_{c=1..C} exp(x_{i,c} - m_i) ) + m_i - x_{i,y_i}
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>
#include <float.h>


__device__ float warp_reduce_sum(float val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float warp_reduce_max(float val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float block_reduce_sum(float val) {
    static __shared__ float shared[32]; // Shared memory for 32 warps
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val); // Each warp performs partial reduction

    if (lane == 0) {
        shared[wid] = val; // Write reduced value to shared memory
    }
    __syncthreads(); // Wait for all warps to finish

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val); // Final reduce within first warp
    }
    return val;
}

__device__ float block_reduce_max(float val){
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_max(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -FLT_MAX;
    if (wid == 0) {
        val = warp_reduce_max(val);
    }
    return val;
}

__global__ void fused_cross_entropy_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ labels,
    float* __restrict__ loss,
    int N,
    int C) {
        // N -> number of samples, C -> number of classes
        int i = blockIdx.x;
        if (i >= N) return;
        const float* row = logits + i * C;
        int target = labels[i];
        __shared__ float shared_max_logit;
        __shared__ float shared_sum_exp;

        // Find the maximum logit for numerical stability
        float max_logit = -FLT_MAX;
        for (int c = threadIdx.x; c < C; c += blockDim.x){
            max_logit = fmaxf(max_logit, row[c]);
        }

        max_logit = block_reduce_max(max_logit);
        if (threadIdx.x == 0) {
            shared_max_logit = max_logit;
        }
        __syncthreads();

        // Compute the sum of exponentials
        float sum_exp = 0.0f;
        for (int c = threadIdx.x; c < C; c += blockDim.x){
            sum_exp += expf(row[c] - shared_max_logit);
        }

        sum_exp = block_reduce_sum(sum_exp);
        if (threadIdx.x == 0) {
            shared_sum_exp = sum_exp;
        }
        __syncthreads();

        if(threadIdx.x == 0){
            float logit_target = row[target];
            float sample_loss = logf(shared_sum_exp) + shared_max_logit - logit_target;
            atomicAdd(loss, sample_loss/N);
        }
}

torch::Tensor fused_cross_entropy_cuda(torch::Tensor logits, torch::Tensor labels) {
    int N = logits.size(0);
    int C = logits.size(1);
    auto loss = torch::zeros({1}, logits.options());
    int threads = 256;
    int blocks = N;
    fused_cross_entropy_kernel<<<blocks, threads>>>(logits.data_ptr<float>(), labels.data_ptr<int64_t>(), loss.data_ptr<float>(), N, C);
    cudaDeviceSynchronize();
    return loss;
}

PYBIND11_MODULE(fused_cross_entropy_module, m) {
    m.def("calc", &fused_cross_entropy_cuda, "Fused Cross Entropy Kernel (CUDA)");
}

