#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MMUOnCPU.hpp"
#include <cuda_runtime.h>

#include <time.h>


/* -- test for version 1.2 --*/
__global__ void waitForHostAndContinue(struct MMUOnTransfer* pMMUOnTransfer)
{
    // int n = 99321021;
    // int count = 0;
    // for(int i = 0; i < n; i++)
    // {
    //     if(n % i == 0)
    //     {
    //         count++;
    //     }
    // }
    // pMMUOnTransfer -> sizeAllocate[BLOCKSIZE*blockIdx.x + threadIdx.x] = count; 
    //allocate call
    int tid =  threadIdx.x;
    int* a = (int*)allocateThr(8192, pMMUOnTransfer);
    
    // *a = tid + 20;

    // if(tid == 0)printf("a\n");

    // int* b = (int*)allocateThr(10, pMMUOnTransfer);
    // *b = tid + 30;
    // freeThr(a, pMMUOnTransfer);
}

int main() 
{
    /* -- init meta -- */
    // init MemoryManagement
    MemoryManagement* memoryManagement[ALLOCATIONMANAGEMENTTHREADNUMBER];
    struct MMUOnTransfer* pMMUOnTransfer;
    cudaError_t cudaStatus = cudaMallocManaged(&pMMUOnTransfer, sizeof(struct MMUOnTransfer));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    //init MMUOnTransfer
    initAllocationStru(memoryManagement, pMMUOnTransfer);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch the kernel
    waitForHostAndContinue<<<BLOCKNUMBER, BLOCKSIZE, 0, stream>>>(pMMUOnTransfer);

    aLaunchKernel(memoryManagement, pMMUOnTransfer, stream);

    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // Wait for the kernel to complete
    printf("pMMUOnTransfer : %d\n", pMMUOnTransfer -> sizeAllocate[0]);
    
    // Calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time:  %f  milliseconds\n", milliseconds);
    
    cudaFree(pMMUOnTransfer);
}
