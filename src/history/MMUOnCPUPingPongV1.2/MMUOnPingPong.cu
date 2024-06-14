#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MMUOnCPU.hpp"
#include <cuda_runtime.h>

/* 
Version 1.1 
In the host, memoryManagement calls the function, allocateMem(), to allocate memory

Version 1.2
add block sychronization
reconstruct the code. users could call the allocateThr and freeThr to call it.
All the scheduling for allocation and free is in the host.
CPU uses a number of threads to deal with the allocation and free requests from GPU. ALLOCATIONMANAGEMENTTHREADNUMBER is used for defining how many threads.
Supports multi-kernel allocation. For each kernel, the host allocate an independent memory meta data to manage it and a thread to deal with.

compiler command
nvcc -rdc=true MMUOnPingPong.cu MMUOnCPU.cu -lpthread
*/

__global__ void waitForHostAndContinue(struct MMUOnTransfer* pMMUOnTransfer)
{
    //allocate call
    int tid =  threadIdx.x;
    int* a = (int*)allocateThr(10, pMMUOnTransfer);
    
    *a = tid + 20;

    int* b = (int*)allocateThr(10, pMMUOnTransfer);
    *b = tid + 30;
    freeThr(a, pMMUOnTransfer);
}

int main() 
{
    /* -- init meta -- */
    // init MemoryManagement
    MemoryManagement* memoryManagement = new MemoryManagement();
    struct MMUOnTransfer* pMMUOnTransfer;
    cudaMallocManaged(&pMMUOnTransfer, sizeof(MMUOnTransfer));
    //init MMUOnTransfer
    initAllocationStru(memoryManagement, pMMUOnTransfer);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Launch the kernel
    waitForHostAndContinue<<<BLOCKNUMBER, BLOCKSIZE, 0, stream>>>(pMMUOnTransfer);

    aLaunchKernel(memoryManagement, pMMUOnTransfer, stream);

    printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress));
    printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress + 1*PAGE_SIZE));
    printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress + 2*PAGE_SIZE));
    printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 3*PAGE_SIZE));
    printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 4*PAGE_SIZE));
    printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 5*PAGE_SIZE));
    printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 6*PAGE_SIZE));
    printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 7*PAGE_SIZE));
    
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