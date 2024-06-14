#ifndef EXTERNAL_H
#define EXTERNAL_H
/* parameters for memory management with linked lists*/
#define MAX_LINKEDLIST_MEM_SIZE (1024*1024*1024) //1GB
/* parameters for memory management with bitmaps*/
#define MAX_BITMAP_MEM_SIZE (1024*1024*1024) //4GB
#define PAGE_SIZE (1024*4)
#define PAGE_NUMBER  (1024*1024)   // (MAX_BITMAP_MEM_SIZE/PAGE_SIZE) need 128KB
#define BITMAP_SIZE (1024*1024/8)   // (PAGE_NUMBER/8)
#define HASHTABLE_BITMAP_BLOCKIDSIZE_SIZE 4096
#define THRESHOLD_MEMSIZE (1024*1024*32)  //the threshold is 4MB. if greater than it, allocate as linked lists.

// some parameters for optimizaiton
#define ALLOCATIONMANAGEMENTTHREADNUMBER  20 //there are a number of threads for dealing with allocaiton from GPU blocks

//user define
#define BLOCKNUMBER 100
#define BLOCKSIZE 1

/* -- Version 1.3 */
#define PORTION_SIZE 64 // Each portion size in bytes
#define PORTIONS_PER_BLOCK 256 // Number of portions per block
#define TOTAL_PORTIONS (PORTIONS_PER_BLOCK * BLOCKNUMBER)
#define POOL_SIZE (TOTAL_PORTIONS * PORTION_SIZE) // Total pool size in bytes
/* -- End Version 1.3 */

/*-- Version 1.3.2*/
#define MAX_SIZE_STACK_USED_POOL 5 // Define the maximum size of the stack for memory pool that the thread have used
/* -- Version 1.3.2*/



/* -- Define data structure for memory management with linked lists -- */
//units of linked lists in memory management 
typedef struct MemorySegmentNode
{
    unsigned int offsetAddress; // start address in the node expressed as a offset
    unsigned int length; 
    bool isHole; // is the memory block is idle
    MemorySegmentNode* next;
} MemorySegmentNode;

class LinkedListManagement
{
    private:
        MemorySegmentNode* ListHead;
        void* startAddress;  // a pointer pointing to unfied memory allocated
        
        /*————Definitions of Functions */
        //find a suitable segment from the freeList
        MemorySegmentNode* findSuitableMem(unsigned int sizeAllocate);

    public:
        LinkedListManagement(void* startAddressIn);
        ~LinkedListManagement();
        
        //Allocation new memory with LinkedList
        void* allocateMem(unsigned int sizeAllocate);
        //free memory with LinkedList
        void freeMem(void* addressIn);
};
/* -- End -- Define data structure for memory management with linked lists -- */

/* -- Define data structure for memory management with bitmaps*/ 
typedef struct {
    unsigned int blockID;
    unsigned int size;
} BlockIDSize;

typedef struct BlockIDSizeNode {
    BlockIDSize pair;
    struct BlockIDSizeNode* next;
} BlockIDSizeNode;

class BitmapManagement
{
    private:
        void* startAddress; //a pointer pointing to unfied memory allocated
        unsigned char bitmap[BITMAP_SIZE];  // Bitmap to track memory blocks
        BlockIDSizeNode* buckets[HASHTABLE_BITMAP_BLOCKIDSIZE_SIZE];  //used for storing the blockID and their size
        
        /*————Definitions of Functions */
        void initializeBitmap(); // init bitmap
        //set a bit in the bitmap.  This function sets a bit at a given index to indicate the block is used.
        void setBitMapTrue(unsigned int blockIndex);  
        //set continuous bits in the bitmap This function sets continuous bits at a given starting index to indicate the block is used and the size of continuous bits.
        void setBitsMapTrue(unsigned int size, unsigned int startBlockIndex); 
        //This function clears a bit at a given index to indicate the block is free
        void clearBit(unsigned int blockIndex);
        //clears continuous bits at a given index to indicate the block is free and size of continuous bits.
        void clearBits(unsigned int size, unsigned int startBlockIndex);
        //This function checks if a bit at a given index is set
        bool isBitSet(unsigned int blockIndex);
        //Function to check if a bit at a given position in a byte is zero
        bool isZeroBit(unsigned char byte, unsigned int position);
        //
        //search suitable continuous pages
        unsigned int findPages(unsigned int pageSize);
        //search only one idle block in bitmaps
        unsigned int findPage();
        
        /*--hash table functions for bockID*/
        //hash function
        unsigned int hash(unsigned int key);
        //init hash table
        void initializeHashTable();
        //insertion of hash table
        void insertHashTable(unsigned int key, unsigned int value);
        //Find of hash table, and return the size
        unsigned int findHashTable(unsigned int key);
        //Delete of hash table
        void deleteHashTable(unsigned int key);
        //free Entire Hash Table
        void freeHashTable();

    public:
        BitmapManagement(void * startAddressIn);
        ~BitmapManagement();
        
        //Allocation new memory with bitmaps
        void* allocateMem(unsigned int sizeAllocate);
        //free memory with bitmaps
        void freeMem(void* addressIn);
};
/* -- End -- Define data structure for memory management with bitmaps*/

/*-- Define data structure for memory management--*/
class MemoryManagement
{
    private:
        void* bitmapStartAddress;
        void* linkedListStartAddress;
        
        BitmapManagement *bitmapManagement;
        LinkedListManagement *linkedListManagement;
    public:
        MemoryManagement();
        ~MemoryManagement();

        void* getBitmapStartAddress();
        void* getLinkedListStartAddress();

        void allocateMem(struct MMUOnTransfer* pMMUOnTransfer, int bid);
        void freeMem(struct MMUOnTransfer* pMMUOnTransfer, int bid);
};
/*-- End -- Define data structure for memory management--*/


/* --  Define data structure for Transferring between CPU and GPU--*/
typedef struct MMUOnTransfer
{
    // void* bitmapStartAddress;
    // void* linkedListStartAddress;
    int syncFlag[BLOCKNUMBER];
    unsigned int sizeAllocate[BLOCKNUMBER*BLOCKSIZE];
    void* addressAllocate[BLOCKNUMBER*BLOCKSIZE];
    void* addressFree[BLOCKNUMBER*BLOCKSIZE];
} MMUOnTransfer;

/* -- End --  Define data structure for Transferring between CPU and GPU--*/

//a struct message for a allocation thread arg
typedef struct {
    MemoryManagement *pmemoryManagement[ALLOCATIONMANAGEMENTTHREADNUMBER];
    struct MMUOnTransfer *pMMUOnTransfer;
    int *should_exit; // Flag to indicate the thread should exit
} thread_args;

//a struct message for deal with some block allocations
typedef struct {
    // MemoryManagement *pmemoryManagement;
    // struct MMUOnTransfer *pMMUOnTransfer;
    int start;
    int end;
    // volatile int should_exit; // Flag to indicate the thread should exit
    struct MMUOnTransfer *pMMUOnTransfer;
    int *should_exit; // Flag to indicate the thread should exit
    MemoryManagement *pmemoryManagement;
} threadBlockAllocations;

/* -- Version 1.3.2*/
//Stack structure for stored the memory portions that thread has used
typedef struct {
    int items[MAX_SIZE_STACK_USED_POOL];
    int top;
} StackUsedPool;
/* -- End -- Version 1.3.2*/

/* -- allocation function*/
//allocation function as a thread
void* threadAllocation(void* arg);
//each thread deal with block allocations
void* blockAllocationThr(void* arg);
//after launch kernel
void aLaunchKernel(thread_args* args, cudaStream_t stream);
//before launch kernel, init it
void initAllocationStru(MemoryManagement* memoryManagement[], struct MMUOnTransfer *pMMUOnTransfer, pthread_t* thread_id, thread_args* args, int* should_exit);
/* -- End -- allocation function*/

/* -- user library in GPU*/
//allocation
__device__ void* allocateThr(size_t allocateSize, struct MMUOnTransfer* pMMUOnTransfer);

//free
__device__ void freeThr(void* freeAddress, struct MMUOnTransfer* pMMUOnTransfer);
/* -- End -- allocation in GPU*/

/* Function in Version 1.3*/
//The host initializes the GPU memory pool.
void initMemoryPoolOnHost();

// Kernel to initialize memory pool and allocation map
__global__ void initMemoryPool();

// Device function to allocate memory portion
__device__ char* allocateMemoryPoolGPU();

// Device function to deallocate memory portion
__device__ void deallocateMemoryPoolGPU(void* ptr);
/* -- End -- Function in Version 1.3*/

/* Function in version 1.3.1*/
__device__ char* allocateMemoryPoolGPURandomly();

__device__ void deallocateMemoryPoolGPURandomly(void* vptr);
/* -- End -- Function in version 1.3.1*/

/* Function in version 1.3.2*/
// Function to initialize the stack in memory pool
__device__ void initStackUsedPool(StackUsedPool *s);

// Function to check if the stack is full
__device__ bool isStackUsedPoolFull(StackUsedPool *s);

// Function to check if the stack is empty
__device__ bool isStackUsedPoolEmpty(StackUsedPool *s);

// Function to add an element to the stack
__device__ void pushStackUsedPool(StackUsedPool *s, int element);

// Function to remove an element from the stack
__device__ int popStackUsedPool(StackUsedPool *s);

//add stack in memory pool
__device__ char* allocateMemoryPoolGPURandomlyStack(StackUsedPool *s);

__device__ void deallocateMemoryPoolGPURandomlyStack(void* vptr, StackUsedPool *s);
/* -- End -- Function in version 1.3.2*/
#endif

