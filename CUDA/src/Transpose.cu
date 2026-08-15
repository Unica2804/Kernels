# include <iostream>
# include <cuda_runtime.h>

__global__ void transpose(const float *in, float *out, size_t N){
    size_t col = threadIdx.x + blockIdx.x * blockDim.x;
    size_t row = threadIdx.y + blockIdx.y * blockDim.y;
    if (row < N && col <N){
        out[col * N +row] = in[row * N +col];
    }
}

int main(){
    size_t N = 1024;
    size_t bytes = N * N * sizeof(float);
    float *d_in;  float *d_out;

    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);

    float *h_in = new float[N * N];
    for (size_t i = 0; i < N * N; i++){
        h_in[i] = i;
    }
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    transpose<<<grid, block>>>(d_in, d_out, N);
    float *h_out = new float[N * N];
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (size_t i = 0; i < N * N; i++) {
        size_t row = i / N;
        size_t col = i % N;
        float expected = static_cast<float>(col * N + row);  // transposed index value

        if (h_out[i] != expected) {
            std::cerr << "Error at index " << i
                      << ": expected " << expected
                      << ", got " << h_out[i] << std::endl;

            cudaFree(d_in);
            cudaFree(d_out);
            delete[] h_in;
            delete[] h_out;
            return -1;
        }
    }
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
    std::cout << "Transpose successful!" << std::endl;
    return 0;        

}