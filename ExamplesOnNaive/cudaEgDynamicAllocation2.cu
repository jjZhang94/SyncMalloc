#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/resource.h>
#include <iostream>


/*
This is an example to illustrate processes in dynamic allocation
and the failure of oversubscription in dynamic allocation
*/

__global__ void dynamicAllocKernel() {
    
    // Dynamic allocation
    size_t size = 123;
    char* ptr = (char*)malloc(size);
    if (ptr == NULL)
    {
        printf("EE");
        return;
    }
    memset(ptr, 0, size);
    printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
    free(ptr);
}

int main() {
    int N = 5; // Size of the array

    // Kernel launch parameters
    int blockSize = 5;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Launch the kernel
    dynamicAllocKernel<<<numBlocks, blockSize>>>();

    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}

