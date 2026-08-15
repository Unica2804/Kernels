#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

#define FULL_MASK 0xffffffff

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            throw std::runtime_error(cudaGetErrorString(err__)); \
        } \
    } while (0)

__device__ float warp_max(float val){
    for(size_t offset = warpSize/2; offset > 0; offset /=2){
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

__device__ float warp_sum(float val){
    for(size_t offset = warpSize/2; offset > 0; offset /=2){
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}


__global__ void softmax(const float *input, float *output, size_t N){
    extern __shared__ float sdata[];
    
    size_t tid = threadIdx.x;
    size_t lane = tid % warpSize;
    size_t wid = tid / warpSize;
    size_t num_warps = blockDim.x / warpSize;

    // warp level max value

    float val = (tid < N) ? input[tid] : -INFINITY;
    float max_val = warp_max(val);

    if(lane == 0){
        sdata[wid] = max_val;
    }
    __syncthreads();

    // global max value

    float global_max = (wid == 0 && lane < num_warps) ? sdata[lane] : -INFINITY;
    if (wid == 0){
        global_max = warp_max(global_max);
    }
    __syncthreads();

    if (tid == 0){
        sdata[0] = global_max;
    }
    __syncthreads();
    global_max = sdata[0];

    float exp_val = (tid < N) ? expf(val - global_max) : 0.0f;
    float sum_exp = warp_sum(exp_val);

    if (lane == 0){
        sdata[wid] = sum_exp;
    }
    __syncthreads();

    float global_sum_exp = (wid == 0 && lane < num_warps) ? sdata[lane] : 0.0f;
    if (wid == 0){
        global_sum_exp = warp_sum(global_sum_exp);
    }

    if (tid == 0){
        sdata[0] = global_sum_exp;
    }
    __syncthreads();
    global_sum_exp = sdata[0];

    if (tid < N){
        output[tid] = exp_val / global_sum_exp;
    }
}

void apply_softmax(const std::vector<float> &input){
    size_t N = input.size();
    float *d_input, *d_output;
    size_t bytes = N * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
    size_t threads_per_block = ((N+31)/32)*32; // round up to nearest multiple of 32
    if (threads_per_block > 1024) {
        threads_per_block = 1024; // max threads per block
    }
    size_t num_warps = threads_per_block / 32;
    size_t shared_mem_size = num_warps * sizeof(float);
    softmax<<<1, threads_per_block, shared_mem_size>>>(d_input, d_output, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> h_output(N);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    float total = 0.0f;
    std::cout << "Softmax Output: ";
    for (size_t i = 0; i < N; ++i) {
        std::cout << h_output[i] << " ";
        total += h_output[i];
    }
    std::cout << "\n\nTotal Sum: " << total << std::endl;
}

int main() {
    // Large values that would usually overflow exp(x)
    std::vector<float> input = {100.0f, 101.0f, 99.0f, 50.0f, 100.5f};
    
    try {
        apply_softmax(input);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}