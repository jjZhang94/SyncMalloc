#include <cuda_runtime.h>
#include <iostream>


__global__ void sumBlockElements(int *blockSum) {
    // Declare shared memory dynamically
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    // unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load each thread's data into shared memory
    sdata[tid] = tid; // Example: using thread index as data; replace with actual data if needed
    __syncthreads(); // Ensure all threads have written their data to shared memory

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); // Make sure all additions at one stage are done!
    }

    // Write result for this block to global memory
    if (tid == 0) blockSum[blockIdx.x] = sdata[0];
}


int main() {
    int numBlocks = 100;
    int blockSize = 1024; // Choose a power of two for simplicity

    int *d_blockSum;
    cudaMalloc(&d_blockSum, numBlocks * sizeof(int));

    // Launch the kernel
    sumBlockElements<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_blockSum);

    // Copy the block sums back to host
    int h_blockSum[numBlocks];
    cudaMemcpy(h_blockSum, d_blockSum, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the sum for each block
    for (int i = 0; i < numBlocks; i++) {
        std::cout << "Sum for block " << i << ": " << h_blockSum[i] << std::endl;
    }

    // Cleanup
    cudaFree(d_blockSum);

    return 0;
}

