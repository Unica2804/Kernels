# include <cuda_runtime.h>
# include <iostream>

__global__ void swiglu(const float *input, float *output, size_t N){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        float gate=input[idx];
        float value=input[idx+N];
        float silu=gate/(1+expf(-gate));
        output[idx]=silu*value;
    }
}

void apply_swiglu(const float *input, float *output, size_t N){
    float *d_input, *d_output;
    size_t input_bytes = 2*N*sizeof(float);
    size_t output_bytes = N*sizeof(float);
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_output, output_bytes);
    cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);
    dim3 blockDim(256);
    dim3 gridDim((N+blockDim.x-1)/blockDim.x);
    swiglu<<<gridDim, blockDim>>>(d_input, d_output, N);
    // cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main(){
    const size_t N = 1000000;
    float *input = new float[2*N]; // First N for gate, next N for value
    float *output = new float[N];
    for(size_t i = 0; i < N; ++i){
        input[i] = static_cast<float>(i) - 100.0f; // Gate values from -500 to +500
        input[i+N] = static_cast<float>(i); // Value from 0 to 999999
    }
    apply_swiglu(input, output, N);
    // Print first 10 results for verification
    for(size_t i = 0; i < 20; ++i){
        std::cout << "SiGLU(" << input[i] << ", " << input[i+N] << ") = " << output[i] << '\n';
    }
    delete[] input;
    delete[] output;
    return 0;
}