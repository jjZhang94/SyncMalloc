#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MMUOnCPU.hpp"
#include <cuda_runtime.h>


/* 
Version 1.1 
In the host, memoryManagement calls the function, allocateMem(), to allocate memory


*/

__global__ void waitForHostAndContinue(struct MMUOnTransfer* pMMUOnTransfer) {
        
    //allocate  
    int tid = threadIdx.x;
    pMMUOnTransfer -> sizeAllocate[tid] = 10;
    atomicSub(&(pMMUOnTransfer -> counterForDoHOSTSync), 1);

    // Wait for the host to set the continue flag
    while (atomicAdd(&(pMMUOnTransfer -> syncFlag), 0) == 0) 
    {
    }

    // Continue with the rest of GPU tasks
    int* a = (int*)(pMMUOnTransfer -> addressAllocate[tid]);
    *a = tid + 20;

    // Indicate that this thread has completed its work

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
    int* b = (int*)(pMMUOnTransfer -> addressAllocate[tid]);
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
        pMMUOnTransfer -> addressAllocate[i] = NULL;
    }
    


    // Launch the kernel
    waitForHostAndContinue<<<1, 2>>>(pMMUOnTransfer);

    // The host code to be executed when the GPU sets the flag
    while (pMMUOnTransfer->counterForDoHOSTSync>0) {
        // Optionally, sleep for a short duration to reduce CPU usage
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    memoryManagement.allocateMem(pMMUOnTransfer);
    pMMUOnTransfer -> syncFlag = 1;
    printf("1 finished\n");

    while (pMMUOnTransfer->counterForDoHOSTSync>0 || pMMUOnTransfer -> syncFlag == 1) {
        // Optionally, sleep for a short duration to reduce CPU usage
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    memoryManagement.allocateMem(pMMUOnTransfer);
    pMMUOnTransfer -> syncFlag = 1;
    printf("2 finished\n");

    // Use cudaDeviceSynchronize() to wait for GPU to finish
    cudaDeviceSynchronize();

    printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress));
    printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress + 1*PAGE_SIZE));
    printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress + 2*PAGE_SIZE));
    printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 3*PAGE_SIZE));
    
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