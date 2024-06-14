#include "MMUOnCPU.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

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

void* MemoryManagement::allocateMem(unsigned int sizeAllocate)
{
    if(sizeAllocate * 8 > THRESHOLD_MEMSIZE)
    {
        return linkedListManagement->allocateMem(sizeAllocate);
    }else
    {
        return bitmapManagement->allocateMem(sizeAllocate);
    }
}

void MemoryManagement::freeMem(void* addressIn)
{
    void* bitmapEndAddress = (void*)((char*)(bitmapStartAddress) + MAX_BITMAP_MEM_SIZE);
    if(addressIn >= bitmapStartAddress && addressIn < bitmapEndAddress)
    {
        bitmapManagement->freeMem(addressIn);
    }else
    {
        linkedListManagement->freeMem(addressIn);
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

/* -- Implementation of data structure for Transferring between CPU and GPU--*/


