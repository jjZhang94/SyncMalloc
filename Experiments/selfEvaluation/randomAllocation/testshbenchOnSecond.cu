#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MMUOnCPU.hpp"
#include <cuda_runtime.h>
#include <time.h>


/* 
generate a new random order with fixed sizes

compiler command
nvcc -rdc=true testshbench.cu MMUOnCPU.cu -lpthread
*/

#define ALL_FREE_TIMES 16000
#define PORTION_PER_THREAD (PORTIONS_PER_BLOCK)

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

__global__ void testRandomly(bool* d_isAllocate, int* d_freeBlock, struct MMUOnTransfer* pMMUOnTransfer)
{
    //init memory pool used stack
    StackUsedPool s;
    initStackUsedPool(&s);

    int* allocationPointers[PORTION_PER_THREAD];
    for(int i = 0; i < ALL_FREE_TIMES; i++)
    {
    //     // if(threadIdx.x == 0)
    //     // {
    //     //     printf("AA-");
    //     // }
        
        if(d_isAllocate[i] == 1)
        {
            allocationPointers[d_freeBlock[i]] = (int*)allocateThr(8192, pMMUOnTransfer);
        }else
        {
            freeThr(allocationPointers[d_freeBlock[i]], pMMUOnTransfer);
        }
        if(threadIdx.x == 0)
        {
            printf("A\n");
        }
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

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //generate a allocate or free sequence in a random order
    bool isAllocate[ALL_FREE_TIMES];
    int freeBlock[ALL_FREE_TIMES];

    //device varaiables
    bool *d_isAllocate;
    int *d_freeBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory on the device
    cudaMalloc(&d_isAllocate, ALL_FREE_TIMES * sizeof(bool));
    cudaMalloc(&d_freeBlock, ALL_FREE_TIMES * sizeof(int));

    //init allocatio sequence list
    generateSeq(isAllocate, freeBlock);
    // for(int i =0; i<ALL_FREE_TIMES; i++)
    // {
    //     printf("%d\n",isAllocate[i]);
    // }

    // Copy the host array to the device
    cudaMemcpy(d_isAllocate, isAllocate, ALL_FREE_TIMES * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_freeBlock, freeBlock, ALL_FREE_TIMES * sizeof(int), cudaMemcpyHostToDevice);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch the kernel
    testRandomly<<<BLOCKNUMBER, BLOCKSIZE, 0, stream>>>(d_isAllocate, d_freeBlock, pMMUOnTransfer);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop); 
    aLaunchKernel(&args, stream);

    // Calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time:  %f  milliseconds\n", milliseconds);
    // std::cout << "Kernel execution time: " << milliseconds << " milliseconds\n";

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free device memory
    cudaFree(d_isAllocate);
    cudaFree(d_freeBlock);
    
    return 0;



    // // Launch the kernel
    // waitForHostAndContinue<<<BLOCKNUMBER, BLOCKSIZE>>>();
}

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
