# include <iostream>
# include <cuda_runtime.h>

__global__ void leaky_relu(const float *input, float *output, size_t N, const float alpha) {

    size_t idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx<N){
        output[idx]=input[idx]>0?input[idx]:alpha*input[idx];
    }
}

void apply_leaky_relu(const float *input, float *output, size_t N, const float alpha) {
    float *d_input, *d_output;
    size_t bytes= N*sizeof(float);
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);
    dim3 blockSize(256);
    dim3 numBlocks((N + blockSize.x - 1) / blockSize.x);
    leaky_relu<<<numBlocks, blockSize>>>(d_input, d_output, N, alpha);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main(){
    const size_t N =10000;
    float *input = new float[N];
    float *output = new float[N];
    for (size_t i=0; i<N; i++){
        input[i] = -(float)i;
    }
    const float alpha = 0.01f;
    apply_leaky_relu(input,output,N,alpha);
    for (size_t i=0; i<10; i++){
        std::cout<<output[i]<<" ";
    }
    std::cout<<std::endl;
    delete[] input;
    delete[] output;
    return 0;
}