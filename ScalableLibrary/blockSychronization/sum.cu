#include <stdio.h>

__global__ void sumNeighborIds(int *output) {
    extern __shared__ int sharedData[];

    int tid = threadIdx.x;
    sharedData[tid] = tid;

    // Wait until all threads in the block have written their IDs to shared memory
    __syncthreads();

    // Compute the sum of a thread's ID and its neighbor's ID
    // For simplicity, assume the block size is even and ignore boundary conditions
    int neighborId = (tid % 2 == 0) ? tid + 1 : tid - 1;
    output[tid] = sharedData[tid] + sharedData[neighborId];
}

int main() {
    const int blockSize = 256; // Example block size
    int *outputHost, *outputDevice;

    // Allocate host and device memory
    outputHost = (int*)malloc(blockSize * sizeof(int));
    cudaMalloc(&outputDevice, blockSize * sizeof(int));

    // Launch the kernel
    sumNeighborIds<<<1, blockSize, blockSize * sizeof(int)>>>(outputDevice);

    // Copy the results back to the host
    cudaMemcpy(outputHost, outputDevice, blockSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < blockSize; i++) {
        printf("Thread %d and its neighbor sum: %d\n", i, outputHost[i]);
    }

    // Free memory
    free(outputHost);
    cudaFree(outputDevice);

    return 0;
}
