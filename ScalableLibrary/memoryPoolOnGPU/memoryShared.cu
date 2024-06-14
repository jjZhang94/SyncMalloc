#include <stdio.h>
#include <cuda_runtime.h>

__device__ int allocateMemoryFromPool(char* pool, int* poolIndex, int portionSize, int maxPortions) {
    int index = atomicAdd(poolIndex, 1);
    if (index < maxPortions) {
        return index * portionSize; // Return the offset in the pool
    } else {
        return -1; // Pool is exhausted
    }
}

__global__ void useMemoryPool(char* globalPool, int* globalPoolIndex, int portionSize, int maxPortionsPerBlock) {
    extern __shared__ int localPoolIndex[]; // Each block has its own pool index
    char* blockPool = globalPool + blockIdx.x * maxPortionsPerBlock * portionSize;
    
    if (threadIdx.x == 0) {
        // Initialize the block's pool index
        localPoolIndex[0] = 0;
    }
    __syncthreads(); // Ensure the pool index is initialized

    // Attempt to allocate memory from the block's pool
    int offset = allocateMemoryFromPool(blockPool, localPoolIndex, portionSize, maxPortionsPerBlock);
    if (offset >= 0) {
        // Memory allocation was successful, the thread can use its portion
        printf("Block %d, Thread %d allocated memory at offset %d\n", blockIdx.x, threadIdx.x, offset);
        
        // Example use of allocated memory
        blockPool[offset] = threadIdx.x; // Just a simple assignment for demonstration
    } else {
        printf("Block %d, Thread %d failed to allocate memory\n", blockIdx.x, threadIdx.x);
    }
}

int main() {
    const int portionSize = 1024; // 1KB per portion
    const int portionsPerBlock = 32; // Example: Each block can allocate memory for 32 portions
    const int nBlocks = 4; // Number of blocks
    const int nThreads = 64; // Number of threads per block
    
    char* globalPool;
    int* globalPoolIndex;
    size_t totalPoolSize = nBlocks * portionsPerBlock * portionSize;
    
    // Allocate the global memory pool and the global pool index
    cudaMalloc(&globalPool, totalPoolSize);
    cudaMalloc(&globalPoolIndex, sizeof(int));
    cudaMemset(globalPool, 0, totalPoolSize);
    cudaMemset(globalPoolIndex, 0, sizeof(int));
    
    // Launch the kernel
    useMemoryPool<<<nBlocks, nThreads, sizeof(int)>>>(globalPool, globalPoolIndex, portionSize, portionsPerBlock);
    
    // Cleanup
    cudaFree(globalPool);
    cudaFree(globalPoolIndex);

    return 0;
}
