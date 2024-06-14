#include <stdio.h>

__device__ void computeNeighborSum(int *sharedData, int *output) {
    int tid = threadIdx.x;
    
    // Ensure all threads have written their IDs to shared memory
    __syncthreads();
    
    // Compute the sum of a thread's ID and its neighbor's ID
    // Handling boundary conditions by wrapping around
    int neighborId = (tid + 1) % blockDim.x;
    output[tid] = sharedData[tid] + sharedData[neighborId];
}

__global__ void sumNeighborIds(int *output) {
    extern __shared__ int sharedData[];

    int tid = threadIdx.x;
    sharedData[tid] = tid;

    // Call the subfunction which includes __syncthreads() and the computation
    computeNeighborSum(sharedData, output);
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
