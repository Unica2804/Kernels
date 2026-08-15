# include <cstdio>
# include <cuda_runtime.h>

__global__ void add(float *a, float *b, float *c, size_t n){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx]=a[idx]+b[idx];
    }
};

int main(){
    const size_t N = 10000000;
    float *a, *b, *c;
    cudaMallocManaged(&a, N*sizeof(float));
    cudaMallocManaged(&b, N*sizeof(float));
    cudaMallocManaged(&c, N*sizeof(float));

    for(size_t i=0; i<N; i++){
        a[i]=1.0f;
        b[i]=2.0f;
    }
    const size_t block_size = 256;
    const size_t grid_size = (N + block_size - 1) / block_size;
    add<<<grid_size, block_size>>>(a,b,c,N);
    cudaGetLastError();
    cudaDeviceSynchronize();
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}