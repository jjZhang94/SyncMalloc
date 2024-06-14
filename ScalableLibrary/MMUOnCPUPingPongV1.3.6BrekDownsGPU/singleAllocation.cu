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


#define ALL_FREE_TIMES 512
#define PORTION_PER_THREAD 10

typedef struct Node {
    int data;
    struct Node* next;
} Node;

// Function prototypes
Node* createNode(int data);

// Function to check if a value exists in the list
int existsInList(Node* head, int data);

// Function to insert a new unique random number into the list
int insertUniqueRandom(Node** head);

// Function to delete a random node from the list
int deleteRandom(Node** head, int length);

void generateSeq(bool* isAllocate, int* freeBlock);


/* -- test for version 1.2 --*/
__global__ void waitForHostAndContinue(struct MMUOnTransfer** pMMUOnTransfer, int *d_sizetmpAllocate, int *d_sizeTotalallocate, bool* d_isAllocate, int* d_freeBlock)
{
    ThreadPointersSizeStore store;
    initializeStoreSecond(&store);

    // int* allocationPointers[PORTION_PER_THREAD];
    int* a = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // if(threadIdx.x == 0)
    // {
    //     printf("step i free %p\n", a);
    // }
    // __syncthreads();
    // int* b = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // __syncthreads();
    // int* c = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // __syncthreads();
    // int* d = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // __syncthreads();
    // int* e = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // __syncthreads();
    // freeThr(d, pMMUOnTransfer, &store, d_sizeTotalallocate, d_sizetmpAllocate);
    // __syncthreads();
    // freeThr(c, pMMUOnTransfer, &store, d_sizeTotalallocate, d_sizetmpAllocate);
    // __syncthreads();
    // freeThr(e, pMMUOnTransfer, &store, d_sizeTotalallocate, d_sizetmpAllocate);
    // __syncthreads();
    // int* f = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // __syncthreads();
    // int* f1 = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // __syncthreads();
    // int* f2 = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // __syncthreads();
    // int* f3 = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // if(threadIdx.x == 0)
    // {
    //     printf("step i free %p\n", f);
    // }
    // freeThr(b, pMMUOnTransfer, &store, d_sizeTotalallocate, d_sizetmpAllocate);
    // __syncthreads();
    // freeThr(c, pMMUOnTransfer, &store, d_sizeTotalallocate, d_sizetmpAllocate);


    // for(int i = 0; i < ALL_FREE_TIMES; i++)
    // {
    //     if(d_isAllocate[i] == 1)
    //     {
    //         allocationPointers[d_freeBlock[i]] = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    //         // if(threadIdx.x == 0 && blockIdx.x == 0)
    //         // {
    //         //     printf("step i allocate %d  %p\n", i, allocationPointers[d_freeBlock[i]]);
    //         // }
    //     }else
    //     {
    //         // deallocateMemoryPoolGPURandomlyStack(allocationPointers[d_freeBlock[i]], &s);
    //         freeThr(allocationPointers[d_freeBlock[i]], pMMUOnTransfer, &store, d_sizeTotalallocate, d_sizetmpAllocate);
    //         // if(threadIdx.x == 0 && blockIdx.x == 0)
    //         // {
    //         //     printf("step i free %d  %p\n", i, allocationPointers[d_freeBlock[i]]);
    //         // }
    //     }
    //     __syncthreads();
    // }
    // //allocate call
    // int tid =  threadIdx.x;
    // int* a = (int*)allocateThr(4, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // // *a = tid + 20;
    // // if(tid == 5)
    // // {
    // //     pMMUOnTransfer -> sizeAllocate[blockIdx.x] = *a;
    // // }
    
    // // if(tid == 0)printf("a\n");
    // // int* b = (int*)allocateThr(10, pMMUOnTransfer);
    // // *b = tid + 30;
    // if(tid != 1)
    // {
    //     freeThr(a, pMMUOnTransfer, &store, d_sizeTotalallocate, d_sizetmpAllocate);
    // }
    
}

int main() 
{
    //generate a allocate or free sequence in a random order
    bool isAllocate[ALL_FREE_TIMES];
    int freeBlock[ALL_FREE_TIMES];

    //device varaiables
    bool *d_isAllocate;
    int *d_freeBlock;

    // Allocate memory on the device
    cudaMalloc(&d_isAllocate, ALL_FREE_TIMES * sizeof(bool));
    cudaMalloc(&d_freeBlock, ALL_FREE_TIMES * sizeof(int));

    //init allocatio sequence list
    generateSeq(isAllocate, freeBlock);

    // Copy the host array to the device
    cudaMemcpy(d_isAllocate, isAllocate, ALL_FREE_TIMES * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_freeBlock, freeBlock, ALL_FREE_TIMES * sizeof(int), cudaMemcpyHostToDevice);

    /* -- init meta -- */
    // init MemoryManagement
    MemoryManagement* memoryManagement[ALLOCATIONMANAGEMENTTHREADNUMBER];
    struct MMUOnTransfer** pMMUOnTransfer;
    struct MMUOnTransfer** d_pMMUOnTransfer;
    pMMUOnTransfer =  (MMUOnTransfer **)malloc(BLOCKNUMBER * sizeof(MMUOnTransfer *));
    pthread_t thread_id;
    thread_args args;
    int should_exit = 0;
    for(int i = 0; i < BLOCKNUMBER; i++)
    {
        cudaError_t cudaStatus = cudaMallocManaged(&(pMMUOnTransfer[i]), sizeof(struct MMUOnTransfer));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(cudaStatus));
            return 1;
        }
    }

    //init MMUOnTransfer  
    initAllocationStru(memoryManagement, pMMUOnTransfer, &thread_id, &args, &should_exit);

    cudaMalloc(&d_pMMUOnTransfer, BLOCKNUMBER * sizeof(MMUOnTransfer *));
    cudaMemcpy(d_pMMUOnTransfer, pMMUOnTransfer, BLOCKNUMBER * sizeof(MMUOnTransfer *), cudaMemcpyHostToDevice);

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
    waitForHostAndContinue<<<BLOCKNUMBER, BLOCKSIZE, BLOCKSIZE*sizeof(int), stream>>>(d_pMMUOnTransfer, d_sizetmpAllocate, d_sizeTotalallocate, d_isAllocate, d_freeBlock);

    // printf("SSSSS\n");
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
    // aLaunchKernel(&args, stream);
    // printf("SSSSsssssS\n");
    // Record the stop event
    // Wait for the kernel to complete

    // Calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time:  %f  milliseconds\n", milliseconds);
    for(int i = 0; i<3; i++)
    {
        printf("pMMUOnTransfer %d\n", (pMMUOnTransfer[i])->sizeAllocate);
    }

    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress + 1*PAGE_SIZE));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress + 2*PAGE_SIZE));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 3*PAGE_SIZE));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 4*PAGE_SIZE));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 5*PAGE_SIZE));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 6*PAGE_SIZE));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 7*PAGE_SIZE));
    
    for(int i = 0; i<BLOCKNUMBER; i++)
    {
        cudaFree(pMMUOnTransfer[i]);
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free device memory
    cudaFree(d_isAllocate);
    cudaFree(d_freeBlock);

    
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

//implementation
Node* createNode(int data) 
{
    Node* newNode = (Node*)malloc(sizeof(Node));
    if (newNode == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    newNode->data = data;
    newNode->next = NULL;
    return newNode;
}

int existsInList(Node* head, int data) 
{
    Node* current = head;
    while (current != NULL) {
        if (current->data == data) {
            return 1; // Data found
        }
        current = current->next;
    }
    return 0; // Data not found
}

int insertUniqueRandom(Node** head) 
{
    while(1)
    {
        int randomData = rand() % PORTION_PER_THREAD; // Random number between 0 and PORTION_PER_THREAD
        if (!existsInList(*head, randomData)) {
            Node* newNode = createNode(randomData);
            newNode->next = *head;
            *head = newNode;
            // printf("insert %d\n", randomData);
            return randomData;
        } 
    }
}

int deleteRandom(Node** head, int length) 
{
    int randomIndex = rand() % length;
    Node* prev = NULL;
    Node* current = *head;

    if (randomIndex == 0) { // Delete the head
        *head = current->next;
        int valueD = current -> data;
        free(current);
        // printf("Deleted %d \n", valueD);
        return valueD;
    }

    for (int i = 0; i < randomIndex; i++) {
        prev = current;
        current = current->next;
    }

    prev->next = current->next;
    int valueD = current -> data;
    free(current);
    // printf("Deleted %d\n", valueD);
    return valueD;
}

void generateSeq(bool* isAllocate, int* freeBlock)
{
    srand(time(NULL));
    Node* head = NULL;
    int nodeLength = 0;

    for(int i = 0; i < ALL_FREE_TIMES; i++)
    {
        //if the allocated list is empty, allocate one.
        if(nodeLength == 0)
        {
            freeBlock[i] = insertUniqueRandom(&head);
            isAllocate[i] = 1;
            nodeLength ++;
            continue;
        }

        //if the allocated list is full, delete one.
        if(nodeLength == PORTION_PER_THREAD)
        {
            isAllocate[i] = 0;
            freeBlock[i] = deleteRandom(&head, nodeLength);
            nodeLength --;
            continue;
        }

        //allocate or free randomly
        int isallocate = rand() % 2;
        //if allocate
        if(isallocate == 1)
        {
            freeBlock[i] = insertUniqueRandom(&head);
            isAllocate[i] = 1;
            nodeLength ++;
        }else
        {
            isAllocate[i] = 0;
            freeBlock[i] = deleteRandom(&head, nodeLength);
            nodeLength --;
        }
    }
}

