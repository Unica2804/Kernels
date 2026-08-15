#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#define FULL_MASK 0xffffffff

#define CHECK_CUDA(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            throw std::runtime_error(cudaGetErrorString(err__));                    \
        }                                                                           \
    } while (0)

__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__device__ float warp_reduce_max(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

__device__ float block_reduce_sum(float val) {
    __shared__ float warp_partials[32];
    int lane = static_cast<int>(threadIdx.x) & 31;
    int wid = static_cast<int>(threadIdx.x) >> 5;
    int warp_count = (static_cast<int>(blockDim.x) + 31) >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        warp_partials[wid] = val;
    }
    __syncthreads();

    val = (wid == 0 && lane < warp_count) ? warp_partials[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__device__ float block_reduce_max(float val) {
    __shared__ float warp_partials[32];
    int lane = static_cast<int>(threadIdx.x) & 31;
    int wid = static_cast<int>(threadIdx.x) >> 5;
    int warp_count = (static_cast<int>(blockDim.x) + 31) >> 5;

    val = warp_reduce_max(val);
    if (lane == 0) {
        warp_partials[wid] = val;
    }
    __syncthreads();

    val = (wid == 0 && lane < warp_count) ? warp_partials[lane] : -INFINITY;
    if (wid == 0) {
        val = warp_reduce_max(val);
    }
    return val;
}

__global__ void softmax_attn(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int seq_len,
    int d) {
    // Shared memory layout: query row [d], then scores/weights [seq_len].
    extern __shared__ float smem[];
    float* shared_q = smem;
    float* shared_scores = smem + d;
    __shared__ float shared_max;
    __shared__ float shared_sum;

    int tid = static_cast<int>(threadIdx.x);
    int q_idx = static_cast<int>(blockIdx.x);
    if (q_idx >= seq_len) {
        return;
    }

    for (int i = tid; i < d; i += static_cast<int>(blockDim.x)) {
        shared_q[i] = Q[q_idx * d + i];
    }
    __syncthreads();

    const float scale = rsqrtf(static_cast<float>(d));
    for (int k_idx = tid; k_idx < seq_len; k_idx += static_cast<int>(blockDim.x)) {
        float dot = 0.0f;
        for (int i = 0; i < d; ++i) {
            dot += shared_q[i] * K[k_idx * d + i];
        }
        shared_scores[k_idx] = dot * scale;
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int i = tid; i < seq_len; i += static_cast<int>(blockDim.x)) {
        local_max = fmaxf(local_max, shared_scores[i]);
    }
    float reduced_max = block_reduce_max(local_max);
    if (tid == 0) {
        shared_max = reduced_max;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += static_cast<int>(blockDim.x)) {
        float w = expf(shared_scores[i] - shared_max);
        shared_scores[i] = w;
        local_sum += w;
    }
    float reduced_sum = block_reduce_sum(local_sum);
    if (tid == 0) {
        shared_sum = reduced_sum;
    }
    __syncthreads();

    for (int out_idx = tid; out_idx < d; out_idx += static_cast<int>(blockDim.x)) {
        float acc = 0.0f;
        for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
            float weight = shared_scores[k_idx] / shared_sum;
            acc += weight * V[k_idx * d + out_idx];
        }
        output[q_idx * d + out_idx] = acc;
    }
}


void softmax_attention_gpu(const std::vector<float>& Q,
                          const std::vector<float>& K,
                          const std::vector<float>& V,
                          std::vector<float>& output,
                          int seq_len,
                          int d) {
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_output = nullptr;
    size_t bytes = static_cast<size_t>(seq_len) * d * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_Q, bytes));
    CHECK_CUDA(cudaMalloc(&d_K, bytes));
    CHECK_CUDA(cudaMalloc(&d_V, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    CHECK_CUDA(cudaMemcpy(d_Q, Q.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, K.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, V.data(), bytes, cudaMemcpyHostToDevice));

    int max_shared = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    size_t shared_mem_bytes = static_cast<size_t>(seq_len + d) * sizeof(float);
    if (shared_mem_bytes > static_cast<size_t>(max_shared)) {
        CHECK_CUDA(cudaFree(d_Q));
        CHECK_CUDA(cudaFree(d_K));
        CHECK_CUDA(cudaFree(d_V));
        CHECK_CUDA(cudaFree(d_output));
        throw std::runtime_error("Shared memory requirement exceeds device limit for this shape");
    }

    int threads = 256;
    dim3 grid(seq_len);
    softmax_attn<<<grid, threads, shared_mem_bytes>>>(d_Q, d_K, d_V, d_output, seq_len, d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    output.resize(static_cast<size_t>(seq_len) * d);
    CHECK_CUDA(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_output));
}

int main() {
    int seq_len = 64;
    int d = 64;
    size_t total = static_cast<size_t>(seq_len) * d;

    std::vector<float> Q(total), K(total), V(total);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < total; ++i) {
        Q[i] = dist(rng);
        K[i] = dist(rng);
        V[i] = dist(rng);
    }

    try {
        std::vector<float> gpu_out;
        softmax_attention_gpu(Q, K, V, gpu_out, seq_len, d);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "First 8 output values of row 0: ";
        for (int i = 0; i < std::min(8, d); ++i) {
            std::cout << gpu_out[i] << " ";
        }
        std::cout << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
