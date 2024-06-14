#include "MMUOnCPU.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include <curand_kernel.h>

#include <unistd.h>

/* -- version 1.3*/
// Memory pool structure
__device__ char memoryPool[POOL_SIZE];
__device__ int allocationMapGPU[TOTAL_PORTIONS]; // 0 = free, 1 = allocated
/* -- End -- version 1.3*/

/* -- version 1.3.1*/
// Global variable for CURAND states
__device__ curandState curandStates[TOTAL_PORTIONS];
/* -- End -- version 1.3.1*/

/* -- Implementation of memory management with linked lists -- */
LinkedListManagement::LinkedListManagement(void * startAddressIn)
{
    //init start address
    startAddress = startAddressIn;

    //create a init blanket Node
    MemorySegmentNode* newMemorySegmentNode = (MemorySegmentNode *)malloc(sizeof(MemorySegmentNode));
    if (newMemorySegmentNode == NULL) {
        fprintf(stderr, "Error allocating memory for list node\n");
        exit(EXIT_FAILURE);
    }
    newMemorySegmentNode->length = MAX_LINKEDLIST_MEM_SIZE;
    newMemorySegmentNode->offsetAddress = 0;
    newMemorySegmentNode->next = NULL;
    newMemorySegmentNode->isHole = true;
    
    ListHead = newMemorySegmentNode;

}

LinkedListManagement::~LinkedListManagement()
{
    //free the entire list of MemorySegmentNode
    MemorySegmentNode* currentNode = ListHead;
    MemorySegmentNode* nextNode;
    while(currentNode != NULL){
        nextNode = currentNode -> next;
        free(currentNode);
        currentNode = nextNode;
    }
    ListHead = NULL;

}

MemorySegmentNode* LinkedListManagement::findSuitableMem(unsigned int sizeAllocate)
{
    //return the current node when finding a suitable node
    MemorySegmentNode* returnNode = NULL;
    MemorySegmentNode* currentNode = ListHead;
    while (currentNode != NULL) {
        if((currentNode -> isHole == true) && (currentNode ->length >= sizeAllocate))
        {
            return currentNode;
        }
        currentNode = currentNode->next;
    }
    return returnNode;
}

void* LinkedListManagement::allocateMem(unsigned int sizeAllocate)
{
    void* returnAddress = NULL;
    MemorySegmentNode* insertNode = NULL;
    //find a suitable position to allocate and return the pointer to the last node.
    insertNode = findSuitableMem(sizeAllocate);
    
    //if cannot find a suitable memory block
    if (insertNode == NULL){
        fprintf(stderr, "Error allocating memory for memory\n");
        exit(EXIT_FAILURE);
    }

    /*------Update memory management nodes--------*/
    returnAddress = (void*)((char*)startAddress + insertNode -> offsetAddress);
    //If the allocated length is equal to memory block existed, just chage the status of memory management node
    //isHole to false.
    if(insertNode->length == sizeAllocate)
    {
        insertNode -> isHole = false;
        return returnAddress;
    }

    //create a new Node
    MemorySegmentNode* newMemorySegmentNode = (MemorySegmentNode *)malloc(sizeof(MemorySegmentNode));
    if (newMemorySegmentNode == NULL) {
        fprintf(stderr, "Error allocating memory for list node\n");
        exit(EXIT_FAILURE);
    }

    newMemorySegmentNode -> length = insertNode -> length - sizeAllocate;
    newMemorySegmentNode -> offsetAddress = insertNode -> offsetAddress + sizeAllocate;
    newMemorySegmentNode -> next = insertNode -> next;
    newMemorySegmentNode -> isHole = true;

    //upadate original nodes
    insertNode -> isHole = false;
    insertNode -> length = sizeAllocate;
    insertNode -> next = newMemorySegmentNode;
    
    return returnAddress;
}

void LinkedListManagement::freeMem(void* addressIn)
{
    /*--the lack of check the avaliable scope of addressIn.-- */
    unsigned int currentAddressOffset = (unsigned int)((char*)addressIn - (char*)startAddress);
    //Find the corresponding node
    MemorySegmentNode* currentNode = ListHead;
    MemorySegmentNode* lastNode = NULL;
    MemorySegmentNode* nextNode = NULL;
    while (currentNode != NULL) {
        if (currentNode->offsetAddress == currentAddressOffset) {
            break;
        }
        lastNode = currentNode;
        currentNode = currentNode->next;
    }

    nextNode = currentNode->next;
    //check the results
    // if (currentNode == NULL){
    //     /*---if cannot find it, return an error----*/
    //     return;
    // }

    /* -- merge the neighbor idle memory blocks and clean the unnecessary management nodes*/
    if(lastNode == NULL)
    {
        if(nextNode == NULL)
        {
            currentNode -> isHole = true;
        }else
        {
            if(nextNode ->isHole == false){
                currentNode -> isHole = true;
            }else{
                currentNode -> isHole = true;
                currentNode -> length = nextNode -> length + currentNode -> length;
                currentNode -> next = nextNode -> next;
                free(nextNode);
            }
        }
        return;
    }

    if(nextNode == NULL)
    {
        if(lastNode -> isHole == true)
        {
            lastNode -> next = nextNode;
            lastNode -> length = lastNode -> length + currentNode -> length;
            free(currentNode);
        }else{
            currentNode -> isHole = true;
        }
        return;
    }

    if(lastNode-> isHole == true)
    {
        lastNode -> next = nextNode;
        lastNode -> length = lastNode -> length + currentNode -> length;
        free(currentNode);
        if(nextNode -> isHole == true)
        {
            lastNode -> next = nextNode -> next;
            lastNode -> length = lastNode -> length + nextNode -> length;
            free(nextNode);
        }
        //else if nextNode's isHole is false, do nothing
    }else
    {
        if(nextNode -> isHole == true)
        {
            currentNode -> isHole = true;
            currentNode -> length = nextNode -> length + currentNode -> length;
            currentNode -> next = nextNode -> next;
            free(nextNode);
        }else{
            currentNode -> isHole = true;
        }
    }

}
/* -- End -- Implementation of memory management with linked lists -- */


/* -- Implementation of memory management with bitmaps -- */
BitmapManagement::BitmapManagement(void * startAddressIn)
{
    //init start address
    startAddress = startAddressIn;

    //init bitmap
    initializeBitmap();

    //init hash tables
    initializeHashTable();
}

BitmapManagement::~BitmapManagement()
{
    freeHashTable();
}

void BitmapManagement::initializeBitmap()
{
    memset(bitmap, 0, sizeof(bitmap));
    return;
}

void BitmapManagement::setBitMapTrue(unsigned int blockIndex)
{
    unsigned int byteIndex = blockIndex / 8;
    unsigned int bitIndex = blockIndex % 8;
    unsigned char mask = 1 << bitIndex;
    bitmap[byteIndex] |= mask;
    return;
}

void BitmapManagement::setBitsMapTrue(unsigned int size, unsigned int startBlockIndex)
{
    unsigned int endBlockIndex = startBlockIndex + size;
    unsigned int startByteIndex = startBlockIndex / 8;
    unsigned int endByteIndex = (endBlockIndex - 1) / 8;

    if (startByteIndex == endByteIndex)
    {
        // If size is no larger than 8 and within a single byte
        unsigned int startBitIndex = startBlockIndex % 8;
        unsigned int endBitIndex = endBlockIndex % 8;
        unsigned char mask = ((0xFF >> (8 - endBitIndex)) & (0xFF << startBitIndex));

        bitmap[startByteIndex] |= mask;
        return;
    }

    for (unsigned int i = startByteIndex; i <= endByteIndex; i++) {
        unsigned char mask;

        if (i == startByteIndex) {
            // Create mask for the first byte
            unsigned int startBitIndex = startBlockIndex % 8;
            mask = (0xFF << startBitIndex);
        } else if (i == endByteIndex) {
            // Create mask for the last byte
            unsigned int endBitIndex = endBlockIndex % 8;
            if(endBitIndex == 0)
                endBitIndex = endBitIndex + 8;
            mask = (0xFF >> (8 - endBitIndex));
        } else {
            // Set all bits in the intermediate bytes
            mask = 0xFF;
        }

        bitmap[i] |= mask;
    }
    return;
}

void BitmapManagement::clearBit(unsigned int blockIndex)
{
    unsigned int byteIndex = blockIndex / 8;
    unsigned int bitIndex = blockIndex % 8;
    unsigned char mask = ~(1 << bitIndex);
    bitmap[byteIndex] &= mask;
}

void BitmapManagement::clearBits(unsigned int size, unsigned int startBlockIndex)
{
    unsigned int endBlockIndex = startBlockIndex + size;
    unsigned int startByteIndex = startBlockIndex / 8;
    unsigned int endByteIndex = (endBlockIndex - 1) / 8;

    if (startByteIndex == endByteIndex)
    {
        // If size is no larger than 8 and within a single byte
        unsigned int startBitIndex = startBlockIndex % 8;
        unsigned int endBitIndex = endBlockIndex % 8;
        unsigned char mask = ~((0xFF >> (8 - endBitIndex)) & (0xFF << startBitIndex));

        bitmap[startByteIndex] &= mask;
        return;
    }

    // Handle multiple bytes
    for (unsigned int i = startByteIndex; i <= endByteIndex; i++) 
    {
        unsigned char mask;
        if (i == startByteIndex) {
            // Create mask for the first byte
            unsigned int startBitIndex = startBlockIndex % 8;
            mask = ~(0xFF << startBitIndex);
        } else if (i == endByteIndex) {
            // Create mask for the last byte
            unsigned int endBitIndex = endBlockIndex % 8;
            if(endBitIndex == 0)
                endBitIndex = endBitIndex + 8;
            mask = ~(0xFF >> (8 - endBitIndex));
        } else {
            // Clear all bits in the intermediate bytes
            mask = 0x00;
        }

        bitmap[i] &= mask;
    }
    return;
}

bool BitmapManagement::isBitSet(unsigned int blockIndex)
{
    unsigned int byteIndex = blockIndex / 8;
    unsigned int bitIndex = blockIndex % 8;
    unsigned char mask = 1 << bitIndex;
    return (bool)(bitmap[byteIndex] & mask) != 0;    
}

bool BitmapManagement::isZeroBit(unsigned char byte, unsigned int position) 
{
    return (byte & (1 << position)) == 0;
}

unsigned int BitmapManagement::findPage()
{
    //Simple sequential search
    for(unsigned int i =0; i < BITMAP_SIZE; i++)
    {
        if (!isBitSet(i))
        {
            return i; // Return the index of the allocated block
        }
    }
    return 0;
}

unsigned int BitmapManagement::findPages(unsigned int pageSize)
{
    //find continuous pages
    //Simple sequential search
    unsigned int count = 0; // Count of continuous zeros
    for (unsigned int i = 0; i < PAGE_NUMBER; ++i) {
        if (isZeroBit(bitmap[i / 8], i % 8)) {
            count++;
            if (count == pageSize) {
                return i - pageSize + 1; // Return the starting index of the sequence
            }
        } else {
            count = 0; // Reset count if a one is found
        }
    }
    return 0; // Return 0 if not found
}

void* BitmapManagement::allocateMem(unsigned int sizeAllocate)
{
    //get the size of continuous pages
    unsigned int pages = ceil(float(sizeAllocate) / PAGE_SIZE);
    
    //if just need only one page, just find a idle block
    if (pages == 1)
    {  
        unsigned int targetBlockIndex = findPage();
        setBitMapTrue(targetBlockIndex);
        insertHashTable(targetBlockIndex, pages);
        return (void*)((char*)startAddress + targetBlockIndex*PAGE_SIZE);
    }

    //more than one pages
    unsigned int startBlockIndex = findPages(pages);
    setBitsMapTrue(pages, startBlockIndex);

    //update block-size table
    insertHashTable(startBlockIndex, pages);

    return (void*)((char*)startAddress + startBlockIndex*PAGE_SIZE);

}

void BitmapManagement::freeMem(void* addressIn)
{
    //get the size of blockID
    unsigned int offsetAddress = (unsigned int)((char*)addressIn - (char*)startAddress);
    unsigned int blockID = offsetAddress / PAGE_SIZE;

    //search the blockID and its size
    unsigned pageNumber = findHashTable(blockID);
    //if just free only one page, just clear the block
    if(pageNumber == 1)
    {
        clearBit(pageNumber);  
    }else
    {
        clearBits(pageNumber, blockID);
    }
    
    deleteHashTable(blockID);
    return;
      
}

unsigned int BitmapManagement::hash(unsigned int key)
{
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = (key >> 16) ^ key;
    return key & (HASHTABLE_BITMAP_BLOCKIDSIZE_SIZE - 1); // TABLE_SIZE is 4096
}

void BitmapManagement::initializeHashTable()
{
    for (int i = 0; i < HASHTABLE_BITMAP_BLOCKIDSIZE_SIZE; ++i) {
        buckets[i] = NULL;
    }
}

void BitmapManagement::insertHashTable(unsigned int key, unsigned int value)
{
    unsigned int idx = hash(key);
    BlockIDSizeNode* newNode = (BlockIDSizeNode*)malloc(sizeof(BlockIDSizeNode));

    newNode->pair.blockID = key;
    newNode->pair.size = value;

    // Insert at the beginning of the chain (bucket)
    newNode->next = buckets[idx];
    buckets[idx] = newNode;
}

unsigned int BitmapManagement::findHashTable(unsigned int key)
{
    unsigned int idx = hash(key);
    unsigned int returnValue;
    BlockIDSizeNode* current = buckets[idx];
    while (current != NULL) {
        if (current->pair.blockID == key)
        {
            returnValue = current->pair.size;
            return returnValue;
        }
        current = current->next;
    }
    return 0;
}

void BitmapManagement::deleteHashTable(unsigned int key)
{
    unsigned int idx = hash(key);
    BlockIDSizeNode *current = buckets[idx];
    BlockIDSizeNode *prev = NULL;
    while (current != NULL) {
        if (current->pair.blockID == key) {
            if (prev == NULL) {
                buckets[idx] = current->next;
            } else {
                prev->next = current->next;
            }
            free(current);
            return;
        }
        prev = current;
        current = current->next;
    }
}

void BitmapManagement::freeHashTable()
{
    for (int i = 0; i < HASHTABLE_BITMAP_BLOCKIDSIZE_SIZE; ++i) {
        BlockIDSizeNode* current = buckets[i];
        while (current != NULL) {
            BlockIDSizeNode* temp = current;
            current = current->next;
            free(temp);
        }
    }
}
/* -- End -- Implementation of memory management with bitmaps -- */


/* -- Implementation of memory management -- */
MemoryManagement::MemoryManagement()
{
    //init startAddress

    // bitmapStartAddress = (void*)malloc(MAX_BITMAP_MEM_SIZE);
    // linkedListStartAddress = (void*)malloc(MAX_LINKEDLIST_MEM_SIZE);

    size_t size = 1024*1024*1024;
    size = size * 4;
    cudaMallocManaged(&bitmapStartAddress, size);
    size = 1024*1024*1024;
    cudaMallocManaged(&linkedListStartAddress, size);

    //init class BitmapManagement and LinkedListManagement
    bitmapManagement = new BitmapManagement(bitmapStartAddress);
    linkedListManagement = new LinkedListManagement(linkedListStartAddress);
}

MemoryManagement::~MemoryManagement()
{
    delete bitmapManagement;
    delete linkedListManagement;

    // free(bitmapStartAddress);
    // free(linkedListStartAddress);

    cudaFree(bitmapStartAddress);
    cudaFree(linkedListStartAddress);
}

void MemoryManagement::allocateMem(struct MMUOnTransfer* pMMUOnTransfer, int bid)
{
    int startBlock = bid * BLOCKSIZE;
    for(int i = 0; i < BLOCKSIZE; i++)
    {
        int tmpi = startBlock + i;
        if(pMMUOnTransfer -> sizeAllocate[tmpi]  * 8 > THRESHOLD_MEMSIZE)
        {
            pMMUOnTransfer -> addressAllocate[tmpi] = linkedListManagement->allocateMem(pMMUOnTransfer -> sizeAllocate[tmpi]);
        }else
        {
            pMMUOnTransfer -> addressAllocate[tmpi] = bitmapManagement->allocateMem(pMMUOnTransfer -> sizeAllocate[tmpi]);
        }
    }
}

void MemoryManagement::freeMem(struct MMUOnTransfer* pMMUOnTransfer, int bid)
{
    void* bitmapEndAddress = (void*)((char*)(bitmapStartAddress) + MAX_BITMAP_MEM_SIZE);
    int startBlock = bid * BLOCKSIZE;
    for(int i = 0; i < BLOCKSIZE; i++)
    {
        int tmpi = startBlock + i;
        if((pMMUOnTransfer -> addressFree[tmpi]) >= bitmapStartAddress && (pMMUOnTransfer -> addressFree[tmpi]) < bitmapEndAddress)
        {
            bitmapManagement->freeMem(pMMUOnTransfer -> addressFree[tmpi]);
        }else
        {
            linkedListManagement->freeMem(pMMUOnTransfer -> addressFree[tmpi]);
        }
    }
}

void* MemoryManagement::getBitmapStartAddress()
{
    return bitmapStartAddress;
}

void* MemoryManagement::getLinkedListStartAddress()
{
    return linkedListStartAddress;
}
/* -- End -- Implementation of memory management -- */

/* -- allocation function*/
//launch threads to deal with blocks allocation and free
void* threadAllocation(void* arg)
{
    thread_args* args = (thread_args*)arg;
    
    //create threads for block allocations
    if(BLOCKNUMBER < ALLOCATIONMANAGEMENTTHREADNUMBER)
    {
        // if the number of threads smaller than blocknumber
        // Create and start threads
        pthread_t threads[BLOCKNUMBER];
        threadBlockAllocations pthreadBlockAllocations[BLOCKNUMBER];
        for(int i = 0; i < BLOCKNUMBER; ++i) 
        {
            pthreadBlockAllocations[i].pMMUOnTransfer = args -> pMMUOnTransfer;
            pthreadBlockAllocations[i].should_exit = args -> should_exit;
            pthreadBlockAllocations[i].pmemoryManagement = args -> pmemoryManagement[i];
            pthreadBlockAllocations[i].start = i;
            pthreadBlockAllocations[i].end = i + 1;

            pthread_create(&threads[i], NULL, blockAllocationThr, &pthreadBlockAllocations[i]);
        }

        for (int i = 0; i < BLOCKNUMBER; ++i) {
            pthread_join(threads[i], NULL);
        }
    }else{
        pthread_t threads[ALLOCATIONMANAGEMENTTHREADNUMBER];
        threadBlockAllocations pthreadBlockAllocations[ALLOCATIONMANAGEMENTTHREADNUMBER];

        for (int i = 0; i < ALLOCATIONMANAGEMENTTHREADNUMBER; ++i) {
            pthreadBlockAllocations[i].pMMUOnTransfer = args -> pMMUOnTransfer;
            pthreadBlockAllocations[i].should_exit = args -> should_exit;
            pthreadBlockAllocations[i].pmemoryManagement = args -> pmemoryManagement[i];
            pthreadBlockAllocations[i].start = i * (BLOCKNUMBER / ALLOCATIONMANAGEMENTTHREADNUMBER);
            pthreadBlockAllocations[i].end = (i + 1) * (BLOCKNUMBER / ALLOCATIONMANAGEMENTTHREADNUMBER);
            if(i == (ALLOCATIONMANAGEMENTTHREADNUMBER - 1))
            {
                pthreadBlockAllocations[i].end = BLOCKNUMBER;
            }
            // printf("start %d. end:%d\n", pthreadBlockAllocations[i].start, pthreadBlockAllocations[i].end);
            pthread_create(&threads[i], NULL, blockAllocationThr, &pthreadBlockAllocations[i]);
        }

        for (int i = 0; i < ALLOCATIONMANAGEMENTTHREADNUMBER; ++i) {
            pthread_join(threads[i], NULL);
        }
    }
    return NULL;

    // while(1)
    // {
    //     while (args->pMMUOnTransfer->syncFlag>0 && !args->should_exit) {
    //     }
    //     if(args->should_exit)break;
        
    //     //allocating calculate
    //     args->pmemoryManagement->allocateMem(args->pMMUOnTransfer);
        
    //     //after finishing, notify GPU to continue
    //     args->pMMUOnTransfer -> syncFlag = 1;
    // }
    // pthread_exit(NULL);
}

//each thread deal with block allocations
void* blockAllocationThr(void* arg)
{
    threadBlockAllocations* data = (threadBlockAllocations*)arg;
    
    while (true) {
        for (int i = data->start; i < data->end; ++i) {
            //chek if there are allocation
            if (data->pMMUOnTransfer->syncFlag[i] == 0){
                //allocating calculate
                // printf("bb\n");
                data -> pmemoryManagement->allocateMem(data->pMMUOnTransfer, i);
                // Perform some calculation...
                // printf("aa\n");
                //after finishing, notify GPU to continue
                data -> pMMUOnTransfer -> syncFlag[i] = 1;
            }
            
            //chek if there are free
            if (data->pMMUOnTransfer->syncFlag[i] == 2){
                //allocating calculate
                
                data -> pmemoryManagement->freeMem(data->pMMUOnTransfer, i);
                // Perform some calculation...
                
                //after finishing, notify GPU to continue
                data -> pMMUOnTransfer -> syncFlag[i] = 1;

            }
        }

        //when kernel finishes, checking is over
        if(*(data -> should_exit))
        {
            break;
        }
    }

    pthread_exit(NULL);
}

//after launch kernel
void aLaunchKernel(thread_args* args, cudaStream_t stream)
{
    // //thread launch
    // pthread_t thread_id;

    // // Initialize thread arguments
    // thread_args args;
    //  = { .pMMUOnTransfer = pMMUOnTransfer}; 
    // int ii = 0;
    // args.should_exit = &ii;
    // for(int i = 0; i < ALLOCATIONMANAGEMENTTHREADNUMBER; i++)
    // {
    //     args.pmemoryManagement[i] = memoryManagement[i];
    // }

    // // Create the thread
    // pthread_create(&thread_id, NULL, threadAllocation, &args);
    // if (pthread_create(&thread_id, NULL, threadAllocation, &args)) {
    //     fprintf(stderr, "Error creating thread\n");
    // }

    // Wait for the kernel in this stream to complete
    cudaError_t error = cudaStreamSynchronize(stream);
    
    // Signal the thread to exit
    *(args -> should_exit) = 1;

    // Wait for the thread to finish
    // pthread_join(thread_id, NULL);

    // if (pthread_join(thread_id, NULL)) {
    //     fprintf(stderr, "Error joining thread\n");
    // }

    cudaStreamDestroy(stream);
}

//before launch kernel, init it
void initAllocationStru(MemoryManagement* memoryManagement[], struct MMUOnTransfer *pMMUOnTransfer, pthread_t* thread_id, thread_args* args, int* should_exit)
{
    unsigned int threadNumber = BLOCKNUMBER * BLOCKSIZE;
    // pMMUOnTransfer -> bitmapStartAddress = memoryManagement->getBitmapStartAddress();
    // pMMUOnTransfer -> linkedListStartAddress = memoryManagement->getLinkedListStartAddress();
    
    //create MemoryManagement
    for(int i = 0; i < ALLOCATIONMANAGEMENTTHREADNUMBER; i++)
    {
        memoryManagement[i] = new MemoryManagement();
    }

    for(int i = 0; i < BLOCKNUMBER; i++)
    {
        pMMUOnTransfer -> syncFlag[i] = 1;
    }
    
    for(int i = 0; i < threadNumber; i++)
    {
        pMMUOnTransfer -> sizeAllocate[i] = 0;
        pMMUOnTransfer -> addressAllocate[i] = NULL;
        pMMUOnTransfer -> addressFree[i] = NULL;
    }

    // Initialize thread arguments
    args ->  pMMUOnTransfer = pMMUOnTransfer; 
    args -> should_exit = should_exit;
    for(int i = 0; i < ALLOCATIONMANAGEMENTTHREADNUMBER; i++)
    {
        args -> pmemoryManagement[i] = memoryManagement[i];
    }

    if (pthread_create(thread_id, NULL, threadAllocation, args)) {
        fprintf(stderr, "Error creating thread\n");
    }
}
/* -- End -- allocation function*/

/* -- allocation in GPU*/
//allocation on Each thread
__device__ void* allocateThr(size_t allocateSize, struct MMUOnTransfer* pMMUOnTransfer)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    pMMUOnTransfer -> sizeAllocate[tid + bid * BLOCKSIZE] = allocateSize;
    
    // Ensure all threads have written their allocating size to transfer structure
    __syncthreads();
    
    //after all threads finish send the allocating size to transfer structure,
    //thread 0 sets the flag to zero, to notify CPU to calculate the allocation.
    if(tid == 0)
    {
        // atomicSub(&(pMMUOnTransfer -> syncFlag[bid]), 1);
        //allocation tag
        pMMUOnTransfer -> syncFlag[bid] = 0;
    }
    
    // Ensure threads o have set the flag successfully
    __syncthreads();
    // Wait for the host to finish allocation and set the continue flag
    while (atomicAdd(&(pMMUOnTransfer -> syncFlag[bid]), 0) == 0) 
    {
    }

    //read the allocation adderess
    return pMMUOnTransfer -> addressAllocate[tid + bid * BLOCKSIZE];
}

__device__ void freeThr(void* freeAddress, struct MMUOnTransfer* pMMUOnTransfer)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    pMMUOnTransfer -> addressFree[tid + bid * BLOCKSIZE] = freeAddress;
    
    // Ensure all threads have written their allocating size to transfer structure
    __syncthreads();
    
    //after all threads finish send the allocating size to transfer structure,
    //thread 0 sets the flag to zero, to notify CPU to calculate the allocation.
    if(tid == 0)
    {
        // atomicSub(&(pMMUOnTransfer -> syncFlag[bid]), 1);
        //free tag
        pMMUOnTransfer -> syncFlag[bid] = 2;
    }
    
    // Ensure threads o have set the flag successfully
    __syncthreads();
    // Wait for the host to finish allocation and set the continue flag
    while (atomicAdd(&(pMMUOnTransfer -> syncFlag[bid]), 0) == 0) 
    {
    }

}
/* -- End -- allocation in GPU*/

/* Function in Version 1.3*/
//The host initializes the GPU memory pool.
void initMemoryPoolOnHost()
{
    initMemoryPool<<<BLOCKNUMBER, PORTIONS_PER_BLOCK>>>();
    cudaDeviceSynchronize();
}

__global__ void initMemoryPool()
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < TOTAL_PORTIONS) {
        allocationMapGPU[idx] = 0; // Mark as free

        /* Function in version 1.3.1*/
        //randomly page --  Each thread gets same seed, a different sequence number, no offset
        curand_init(1234, idx, 0, &curandStates[idx]);
    }
}

__device__ char* allocateMemoryPoolGPU()
{
    int blockId = blockIdx.x;
    int startIdx = blockId * PORTIONS_PER_BLOCK;
    for (int i = 0; i < PORTIONS_PER_BLOCK; ++i) {
        if (atomicCAS(&allocationMapGPU[startIdx + i], 0, 1) == 0) {
            // Memory portion allocated
            return &memoryPool[(startIdx + i) * PORTION_SIZE];
        }
    }
    return NULL; // Allocation failed  
}

__device__ void deallocateMemoryPoolGPU(void* vptr)
{
    char *ptr = (char*)vptr;
    int idx = (ptr - memoryPool) / PORTION_SIZE;
    // Mark as free
    atomicExch(&allocationMapGPU[idx], 0);
}
/* -- End -- Function in Version 1.3*/

/* Function in version 1.3.1*/
__device__ char* allocateMemoryPoolGPURandomly()
{
    int blockId = blockIdx.x;
    int startIdx = blockId * PORTIONS_PER_BLOCK;

    while (true) {
        unsigned int randStateIdx = threadIdx.x + blockIdx.x * blockDim.x; // Use thread and block id to access unique curandState
        float fi = curand_uniform(&curandStates[randStateIdx]);
        int i = (int)(fi * PORTIONS_PER_BLOCK);

        // Generate a random portion ID within this block's range
        int portionId = startIdx + i;

        // Try to atomically set the portion's allocation flag from 0 to 1
        if (atomicCAS(&allocationMapGPU[portionId], 0, 1) == 0) {
            // Memory portion successfully allocated
            return &memoryPool[portionId * PORTION_SIZE];
        }
        // If not successful, the loop continues, trying a new random portion ID
    }

    return NULL; // Allocation failed  
}

__device__ void deallocateMemoryPoolGPURandomly(void* vptr)
{
    char *ptr = (char*)vptr;
    int idx = (ptr - memoryPool) / PORTION_SIZE;
    // Mark as free
    atomicExch(&allocationMapGPU[idx], 0);
}
/* -- End -- Function in version 1.3.1*/

/* Function in version 1.3.2*/
__device__ void initStackUsedPool(StackUsedPool *s)
{
    s->top = -1; // Initialize top to -1 indicating the stack is empty
}

__device__ bool isStackUsedPoolFull(StackUsedPool *s) 
{
    return s->top == MAX_SIZE_STACK_USED_POOL - 1;
}

__device__ bool isStackUsedPoolEmpty(StackUsedPool *s) 
{
    return s->top == -1;
}

__device__ void pushStackUsedPool(StackUsedPool *s, int element)
{
    s->items[++s->top] = element; // Increment top and add the element to the stack
}

__device__ int popStackUsedPool(StackUsedPool *s)
{
    return s->items[s->top--]; // Return the top element and decrement top
}

__device__ char* allocateMemoryPoolGPURandomlyStack(StackUsedPool *s)
{
    int blockId = blockIdx.x;
    int startIdx = blockId * PORTIONS_PER_BLOCK;

    //if the stack is not empty, fetch a index from the stack.
    if(isStackUsedPoolEmpty(s) == 0)
    {
        for(int i = 0; i < MAX_SIZE_STACK_USED_POOL; i++)
        {
            int si = popStackUsedPool(s);
            if (atomicCAS(&allocationMapGPU[si], 0, 1) == 0) 
            {
                // Memory portion successfully allocated
                return &memoryPool[si * PORTION_SIZE];
            }
        }
    }

    while (true) {
        unsigned int randStateIdx = threadIdx.x + blockIdx.x * blockDim.x; // Use thread and block id to access unique curandState
        float fi = curand_uniform(&curandStates[randStateIdx]);
        int i = (int)(fi * PORTIONS_PER_BLOCK);

        // Generate a random portion ID within this block's range
        int portionId = startIdx + i;

        // Try to atomically set the portion's allocation flag from 0 to 1
        if (atomicCAS(&allocationMapGPU[portionId], 0, 1) == 0) {
            // Memory portion successfully allocated
            return &memoryPool[portionId * PORTION_SIZE];
        }
        // If not successful, the loop continues, trying a new random portion ID
    }

    return NULL; // Allocation failed  
}

__device__ void deallocateMemoryPoolGPURandomlyStack(void* vptr, StackUsedPool *s)
{
    char *ptr = (char*)vptr;
    int idx = (ptr - memoryPool) / PORTION_SIZE;

    if(isStackUsedPoolFull(s) == 0)
    {
        pushStackUsedPool(s, idx);
    }

    // Mark as free
    atomicExch(&allocationMapGPU[idx], 0);
}
/* -- End -- Function in version 1.3.1*/