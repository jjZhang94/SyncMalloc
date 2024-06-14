#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MMUOnCPU.hpp"
#include <cuda_runtime.h>

__global__ void waitForHostAndContinue(struct MMUOnTransfer* pMMUOnTransfer, int* doneCounter) {
        
    //allocate  - 
    int tid = threadIdx.x;
    pMMUOnTransfer -> sizeAllocate[tid] = 10;
    atomicSub(&(pMMUOnTransfer -> counterForDoHOSTSync), 1);

    // Wait for the host to set the continue flag
    while (atomicAdd(&(pMMUOnTransfer -> syncFlag), 0) == 0) 
    {
    }

    // Continue with the rest of GPU tasks
    int* a = (int*)(char*)(pMMUOnTransfer -> bitmapStartAddress + pMMUOnTransfer -> offset[tid]);
    *a = tid + 20;

    // Indicate that this thread has completed its work
    // atomicAdd(doneCounter, 1);
    atomicAdd(&(pMMUOnTransfer -> counterForDoHOSTSync), 1);

    // Reset flag for the next round
    // Ensure all threads have finished before resetting
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        while (atomicAdd(&(pMMUOnTransfer -> counterForDoHOSTSync), 0) != pMMUOnTransfer -> threadNumber) {}
        pMMUOnTransfer -> syncFlag = 0; 
    }

    //wait thread 0 set flag 1
    while (atomicAdd(&(pMMUOnTransfer -> syncFlag), 0) != 0) 
    {
    }

    //allocate twice
    pMMUOnTransfer -> sizeAllocate[tid] = 20;
    atomicSub(&(pMMUOnTransfer -> counterForDoHOSTSync), 1);

    // Wait for the host to set the continue flag
    while (atomicAdd(&(pMMUOnTransfer -> syncFlag), 0) == 0) 
    {
    }

    // Continue with the rest of GPU tasks
    int* b = (int*)(char*)(pMMUOnTransfer -> bitmapStartAddress + pMMUOnTransfer -> offset[tid]);
    *b = tid + 30;

}

int main() 
{
    /* -- init meta -- */
    unsigned int threadNumber = 2;
    // init MemoryManagement
    MemoryManagement memoryManagement;
    struct MMUOnTransfer *pMMUOnTransfer;
    cudaMallocManaged(&pMMUOnTransfer, sizeof(MMUOnTransfer));
    //init MMUOnTransfer
    pMMUOnTransfer -> bitmapStartAddress = memoryManagement.getBitmapStartAddress();
    pMMUOnTransfer -> linkedListStartAddress = memoryManagement.getLinkedListStartAddress();
    pMMUOnTransfer -> threadNumber = threadNumber;
    pMMUOnTransfer -> syncFlag = 0;
    pMMUOnTransfer -> counterForDoHOSTSync = threadNumber;
    for(int i = 0; i < threadNumber; i++)
    {
        pMMUOnTransfer -> sizeAllocate[i] = 0;
        pMMUOnTransfer -> offset[i] = 0;
    }
    
    int* doneCounter;
    cudaMallocManaged(&doneCounter, sizeof(int));
    *doneCounter = 0;    


    // Launch the kernel
    waitForHostAndContinue<<<1, 2>>>(pMMUOnTransfer, doneCounter);
    //lanch kernels

    // The host code to be executed when the GPU sets the flag
    while (pMMUOnTransfer->counterForDoHOSTSync>0) {
        // Optionally, sleep for a short duration to reduce CPU usage
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // memoryManagement.allocate(pMMUOnTransfer -> sizeAllocate);
    // memoryManagement.allocate(pMMUOnTransfer -> sizeAllocate);
    pMMUOnTransfer -> offset[0] = 64;
    pMMUOnTransfer -> offset[1] = 128;
    pMMUOnTransfer -> syncFlag = 1;
    printf("1 finished\n");

    while (pMMUOnTransfer->counterForDoHOSTSync>0 || pMMUOnTransfer -> syncFlag == 1) {
        // Optionally, sleep for a short duration to reduce CPU usage
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    pMMUOnTransfer -> offset[0] = 256;
    pMMUOnTransfer -> offset[1] = 512;
    pMMUOnTransfer -> syncFlag = 1;
    printf("2 finished\n");

    // Use cudaDeviceSynchronize() to wait for GPU to finish
    cudaDeviceSynchronize();

    printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress + 64));
    printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress + 128));
    printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress + 256));
    printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 512));
    
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