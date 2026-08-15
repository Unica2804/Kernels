# include <iostream>
# include <cuda_runtime.h>

__global__ void relu(const float *input, float *output, size_t N){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i<N){
        output[i] = input[i]>0?input[i]:0;
    }
}

void apply_relu(const float *input, float *output, size_t N){
    float *d_input, *d_output;
    size_t total_bytes = N*sizeof(float);
    cudaMalloc((void**)&d_input, total_bytes);
    cudaMalloc((void**)&d_output, total_bytes);
    cudaMemcpy(d_input, input, total_bytes, cudaMemcpyHostToDevice);
    int block_size = 256;
    int num_blocks = (N+block_size-1)/block_size;
    relu<<<num_blocks,block_size>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, total_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main(){
    size_t N =2048;
    float *input = new float[N];
    float *output = new float[N];
    for (size_t i=0; i<N; i++){
        input[i] = (i%2==0)?(float)i:-(float)i;
    }
    apply_relu(input, output, N);

    std::cout<<"First 10 elements of Input: ";
    for (size_t i=0; i<10; i++){
        std::cout<<input[i]<<" ";
    }
    std::cout<< '\n';
    std::cout<<"First 10 elements of Output: ";
    for (size_t i=0; i<10; i++){
        std::cout<<output[i]<<" ";
    }
    std::cout<< '\n';
    delete[] input;
    delete[] output;
    return 0;
}