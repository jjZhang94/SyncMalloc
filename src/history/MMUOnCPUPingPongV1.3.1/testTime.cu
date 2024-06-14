#include <cuda_runtime.h>
#include <iostream>

__global__ void exampleKernel(int *data, int n) {
    // Example kernel logic
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1;
    }
}

int main() {
    int n = 1 << 20; // Example array size
    int *d_data;
    size_t bytes = n * sizeof(int);

    // Allocate device memory
    cudaMalloc(&d_data, bytes);

    // Kernel configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Launch the kernel
    exampleKernel<<<numBlocks, blockSize>>>(d_data, n);

    // Record stop event
    cudaEventRecord(stop);

    // Wait for the kernel to complete
    cudaEventSynchronize(stop);

    // Calculate the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution time: " << milliseconds << " milliseconds\n";

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);

    return 0;
}
