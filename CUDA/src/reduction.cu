# include <iostream>
# include <cuda_runtime.h>

__global__ void reduce(float *input, float *output, size_t N){
    extern __shared__ float sdata[];

    size_t tid = threadIdx.x;
    size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
    sdata[tid] = (idx<N) ? input[idx]:0.0f;
    __syncthreads();

    for (size_t s=blockDim.x/2; s>0; s>>=1){
        if (tid<s){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    if (tid==0){
        atomicAdd(output, sdata[0]);
    }
}

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess){
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void apply_reduction(float *input, float *output, size_t N){
    float *d_input, *d_output;
    size_t input_bytes = N*sizeof(float);
    size_t output_bytes = sizeof(float);
    cudaCheckError(cudaMalloc(&d_input, input_bytes));
    cudaCheckError(cudaMalloc(&d_output, output_bytes));
    cudaCheckError(cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemset(d_output, 0, output_bytes));
    size_t block_size = 512;
    size_t grid_size = (N + block_size - 1) / block_size;
    reduce<<<grid_size, block_size, block_size * sizeof(float)>>>(d_input, d_output, N);
    cudaCheckError(cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost));
    cudaCheckError(cudaFree(d_input));
    cudaCheckError(cudaFree(d_output));
}

int main(){
    size_t N = 1<<20; 
    float *h_input = new float[N];
    for (size_t i=0; i<N; ++i){
        h_input[i] = 1.0f; 
    }
    float h_output = 0.0f;
    apply_reduction(h_input, &h_output, N);
    std::cout << "Sum: " << h_output << std::endl; 
    delete[] h_input;
    return 0;
}