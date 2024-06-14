#include <cuda_runtime.h>
#include <iostream>

#define blockSize 1024

__global__ void sumThreadIDs(int* blockSums, int* d_sumtpe) {
    // extern __shared__ int sdata[];

    // Each thread loads its thread ID into shared memory
     int tid = threadIdx.x + blockSize * blockIdx.x;
     int tid1 = threadIdx.x;
    d_sumtpe[tid] = tid1;
    __syncthreads(); // Wait for all threads to load their IDs

    // Perform simple reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid1 < s) {
            d_sumtpe[tid] += d_sumtpe[tid + s];
        }
        __syncthreads(); // Make sure all additions are finished
    }

    // Thread 0 writes the result for this block to global memory
    if (tid1 == 0) {
        blockSums[blockIdx.x] = d_sumtpe[tid];
    }
}


int main() {
    const int numBlocks = 32;
    // const int blockSize = 1024; // Must be a power of 2 for this reduction example
    int* d_blockSums;

    // Allocate memory on the device to store the sum for each block
    cudaMalloc(&d_blockSums, numBlocks * sizeof(int));

    // Calculate shared memory size required for storing thread IDs
    // size_t sharedMemSize = blockSize * sizeof(int);
    int* d_sumtpe;
    cudaMalloc(&d_sumtpe, blockSize * sizeof(int) * numBlocks);

    // Launch the kernel
    sumThreadIDs<<<numBlocks, blockSize>>>(d_blockSums, d_sumtpe);

    // Copy the results back to the host
    int h_blockSums[numBlocks];
    cudaMemcpy(h_blockSums, d_blockSums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the sum of thread IDs for each block
    for (int i = 0; i < numBlocks; i++) {
        std::cout << "Sum of thread IDs in block " << i << ": " << h_blockSums[i] << std::endl;
    }

    // Cleanup
    cudaFree(d_blockSums);

    return 0;
}
