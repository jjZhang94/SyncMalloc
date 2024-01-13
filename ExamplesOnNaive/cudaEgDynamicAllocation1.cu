#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/resource.h>
#include <iostream>

#define NUMBERPERTHREAD 1024  //how many numbers each thread is to deal with.
/*
This is an example to illustrate processes in dynamic allocation
and the failure of oversubscription in dynamic allocation
*/

__global__ void dynamicAllocKernel(int numberThread) {
    // Dynamic allocation
    float* array = (float*)malloc(numberThread * sizeof(float));
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (array == NULL)
    {
        if(idx == 0)
        {
            printf("EE");
        }
        return;
    }


    unsigned long long int start = clock64();
    while (clock64() < start + 1000000) {
        // Busy-wait
    }
    
    

    //printf("Thread %d got pointer: %p\n", idx, array);

    // Free the allocated memory
    free(array);
}

int main() {
    int N = 1024*512; // Size of the threads

    int numberThread = NUMBERPERTHREAD;

    // Kernel launch parameters
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 512*1024*1024);

    // Launch the kernel
    dynamicAllocKernel<<<numBlocks, blockSize>>>(numberThread);

    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}

