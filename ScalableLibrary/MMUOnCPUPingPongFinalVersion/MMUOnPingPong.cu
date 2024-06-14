#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MMUOnCPU.hpp"
#include <cuda_runtime.h>

#include <time.h>

/* 
Version 1.1 
In the host, memoryManagement calls the function, allocateMem(), to allocate memory

Version 1.2
add block sychronization
reconstruct the code. users could call the allocateThr and freeThr to call it.
All the scheduling for allocation and free is in the host.
CPU use a number of thread to deal with the allocation and free requests from GPU. ALLOCATIONMANAGEMENTTHREADNUMBER is used for defining how many threads.
Supports multi-kernel allocation. For each kernel, the host allocate an independent memory meta data to manage it and a thread to deal with.

compiler command
nvcc -rdc=true MMUOnPingPong.cu MMUOnCPU.cu -lpthread

Version 1.3
add memory pool allocation for samll allocation.

Details: 
Implementing a memory pool in CUDA, where each block manages a portion of the memory for dynamic allocation by its threads, with concurrency and atomic operations to manage memory safely. In this example, each CUDA block manages 256 portions of memory, each 64 bytes in size, stored in global memory.

For each thread check the free blocks in a highly parallel manner, the array which tracks the next free portion for each block is stored as bool array, instead of bits.
Since each element would be checked in parallel.

version 1.3.1
Details:
threads search for free pages randomly keeping from heave sychronization.

add a test:
Each thread repeatedly allocates and frees a number of randomly-sized blocks in random order

version 1.3.2
an optimizaiton of allocateMemoryPoolGPURandomly()
Details:
When free the memory from the thread, the memory portions would not be instantsly free to the block.

version 1.3.3
Details:
add test for CPU-GPU kernel
*/

/* -- test for version 1.2 --*/
__global__ void waitForHostAndContinue(struct MMUOnTransfer* pMMUOnTransfer, int *d_sizetmpAllocate, int *d_sizeTotalallocate)
{
    ThreadPointersSizeStore store;
    initializeStoreSecond(&store);

    //allocate call
    int tid =  threadIdx.x;
    int* a = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // *a = tid + 20;
    // if(tid == 5)
    // {
    //     pMMUOnTransfer -> sizeAllocate[blockIdx.x] = *a;
    // }
    
    // if(tid == 0)printf("a\n");
    // int* b = (int*)allocateThr(10, pMMUOnTransfer);
    // *b = tid + 30;
    if(tid != 1)
    {
        freeThr(a, pMMUOnTransfer, &store, d_sizeTotalallocate, d_sizetmpAllocate);
    }
    
}

int main() 
{
    /* -- init meta -- */
    // init MemoryManagement
    MemoryManagement* memoryManagement[ALLOCATIONMANAGEMENTTHREADNUMBER];
    struct MMUOnTransfer* pMMUOnTransfer;
    pthread_t thread_id;
    thread_args args;
    int should_exit = 0;
    cudaError_t cudaStatus = cudaMallocManaged(&pMMUOnTransfer, sizeof(struct MMUOnTransfer));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    //init MMUOnTransfer
    initAllocationStru(memoryManagement, pMMUOnTransfer, &thread_id, &args, &should_exit);

    int* d_sizetmpAllocate;
    cudaMalloc(&d_sizetmpAllocate, sizeof(int)*BLOCKNUMBER*BLOCKSIZE);
    int h_sizetmpAllocate[BLOCKNUMBER*BLOCKSIZE];
    for(int i = 0; i < BLOCKNUMBER*BLOCKSIZE; i++)
    {
        h_sizetmpAllocate[i] = 0;
    }
    cudaMemcpy(d_sizetmpAllocate, &h_sizetmpAllocate, sizeof(int)*BLOCKNUMBER*BLOCKSIZE, cudaMemcpyHostToDevice);

    int* d_sizeTotalallocate;
    cudaMalloc(&d_sizeTotalallocate, sizeof(int)*BLOCKNUMBER*MAX_SECOND_ALLOCATE);
    int h_sizeTotalallocate[BLOCKNUMBER*MAX_SECOND_ALLOCATE];
    for(int i = 0; i < BLOCKNUMBER*MAX_SECOND_ALLOCATE; i++)
    {
        h_sizeTotalallocate[i] = 0;
    }
    cudaMemcpy(d_sizeTotalallocate, &h_sizeTotalallocate, sizeof(int)*BLOCKNUMBER*MAX_SECOND_ALLOCATE, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);
    // Launch the kernel
    waitForHostAndContinue<<<BLOCKNUMBER, BLOCKSIZE, BLOCKSIZE*sizeof(int), stream>>>(pMMUOnTransfer, d_sizetmpAllocate, d_sizeTotalallocate);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop); 
    aLaunchKernel(&args, stream);

    // Record the stop event
    // Wait for the kernel to complete
    
    // Calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time:  %f  milliseconds\n", milliseconds);
    for(int i = 0; i<3; i++)
    {
        printf("pMMUOnTransfer %d\n", pMMUOnTransfer->sizeAllocate[i]);
    }

    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress + 1*PAGE_SIZE));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress + 2*PAGE_SIZE));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 3*PAGE_SIZE));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 4*PAGE_SIZE));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 5*PAGE_SIZE));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 6*PAGE_SIZE));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 7*PAGE_SIZE));
    
    cudaFree(pMMUOnTransfer);

    
    // int* a = (int*)memoryManagement.allocateMem(sizeof(int)*1024*2);
    // int* b = (int*)memoryManagement.allocateMem(sizeof(int));
    // int* c = (int*)memoryManagement.allocateMem(sizeof(int)*1024*4);
    // memoryManagement.freeMem(a);
    // memoryManagement.freeMem(b);
    // memoryManagement.freeMem(c);
    // int* d = (int*)memoryManagement.allocateMem(sizeof(int)*1024*4);
    // memoryManagement.freeMem(d);

    // int* e = (int*)memoryManagement.allocateMem(sizeof(int)*1024*1024*8);
    
    // int* f = (int*)memoryManagement.allocateMem(sizeof(int)*1024*1024*10);
    // int* g = (int*)memoryManagement.allocateMem(sizeof(int)*1024*1024*16);
    // memoryManagement.freeMem(f);
    // int* h = (int*)memoryManagement.allocateMem(sizeof(int)*1024*1024*16);
    // int* i = (int*)memoryManagement.allocateMem(sizeof(int)*1024*1024*9);

    
    // memoryManagement.freeMem(g);
    
    // memoryManagement.freeMem(h);
    
    // memoryManagement.freeMem(e);
    // memoryManagement.freeMem(i);

}
/* -- End test for version 1.2*/

/* -- test for version 1.3*/
// __global__ void waitForHostAndContinue()
// {
//     int tid =  threadIdx.x;
//     int* a = (int*)allocateMemoryPoolGPU();
    
//     *a = tid + 20;

//     int* b = (int*)allocateMemoryPoolGPU();
//     *b = tid + 30;
    
//     deallocateMemoryPoolGPU(a);
// }

// int main() 
// {
//     /* -- init meta -- */
//     // init MemoryManagement
//     initMemoryPoolOnHost();

//     // Launch the kernel
//     waitForHostAndContinue<<<BLOCKNUMBER, BLOCKSIZE>>>();
// }
/* -- End test for version 1.3*/