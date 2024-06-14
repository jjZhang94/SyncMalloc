#include <curand_kernel.h>

// Global variable for CURAND states
__device__ curandState curandStates[TOTAL_PORTIONS];

// Kernel to initialize CURAND states
__global__ void initCurandStates() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < TOTAL_PORTIONS) {
        // Each thread gets same seed, a different sequence number, no offset
        curand_init(1234, idx, 0, &curandStates[idx]);
    }
}

// Modified allocateMemory function
__device__ char* allocateMemory(int blockId) {
    int startIdx = blockId * PORTIONS_PER_BLOCK;
    int portionId;
    int limit = startIdx + PORTIONS_PER_BLOCK;
    unsigned int randStateIdx = threadIdx.x + blockIdx.x * blockDim.x; // Use thread and block id to access unique curandState

    // Loop until we successfully allocate a portion or run out of options
    while (true) {
        // Generate a random portion ID within this block's range
        portionId = startIdx + curand(&curandStates[randStateIdx]) % PORTIONS_PER_BLOCK;

        // Try to atomically set the portion's allocation flag from 0 to 1
        if (atomicCAS(&allocationMap[portionId], 0, 1) == 0) {
            // Memory portion successfully allocated
            return &memoryPool[portionId * PORTION_SIZE];
        }
        // If not successful, the loop continues, trying a new random portion ID
    }
}

// Ensure to call initCurandStates kernel before using allocateMemory
// initCurandStates<<<(TOTAL_PORTIONS + 255) / 256, 256>>>();
