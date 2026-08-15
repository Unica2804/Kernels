// Sparse Matrix-Vector Multiplication (SpMV) in CUDA
#include <iostream>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Warp reduction Helper Function
__device__ inline float warpReduceSum(float val){
    for (int offset = 32/2; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel for SpMV using CSR format
__global__ void spmv_csr_kernel(const float* __restrict__ A, const float* __restrict__ v,
    float* __restrict__ y, int M, int N){
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;

        int row = warp_id;
        if (row < M){
            float partial_sum = 0.0f;
            for (int col = lane_id; col < N; col += 32){
                float val = A[row * N + col];
                if (val != 0.0f) {
                    partial_sum += val * v[col];
                }
            }
            partial_sum = warpReduceSum(partial_sum);

            if (lane_id == 0){
                y[row] = partial_sum;
            }
        }
    }

// Host function to perform SpMV
torch::Tensor spmv_csr(const torch::Tensor& A, const torch::Tensor& v){
    int M = A.size(0);
    int N = A.size(1);

    auto y = torch::zeros({M}, torch::dtype(torch::kFloat32).device(A.device()));

    int threads_per_block = 128;
    int num_blocks = (M + threads_per_block - 1) / threads_per_block;

    spmv_csr_kernel<<<num_blocks, threads_per_block>>>(A.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(), M, N);

    cudaDeviceSynchronize();

    return y;
}

PYBIND11_MODULE(spmv_module, m) {
    m.def("spmv_csr", &spmv_csr, "SpMV CSR Kernel (CUDA)");
}