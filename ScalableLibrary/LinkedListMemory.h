#ifndef EXTERNAL_H
#define EXTERNAL_H
/* parameters for memory management with linked lists*/
#define MAX_LINKEDLIST_MEM_SIZE 1024*1024*1024
/* parameters for memory management with bitmaps*/
#define MAX_BITMAP_MEM_SIZE 1024*1024*1024*4
#define PAGE_SIZE 1024*4
#define PAGE_NUMBER 1024*1024 // need 128KB
#define BITMAP_SIZE 1024*1024/8
#define HASHTABLE_BITMAP_BLOCKIDSIZE_SIZE 4096

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
        void setBitsMapTrue(size_t size, unsigned int startBlockIndex); 
        //This function clears a bit at a given index to indicate the block is free
        void clearBit(unsigned int blockIndex);
        //clears continuous bits at a given index to indicate the block is free and size of continuous bits.
        void clearBits(size_t size, unsigned int startBlockIndex);
        //This function checks if a bit at a given index is set
        bool isBitSet(unsigned int blockIndex);
        //search suitable continuous pages
        unsigned int findPages(size_t pageSize);
        //search only one idle block in bitmaps
        unsigned int findpage();
        
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
        BitmapManagement();
        ~BitmapManagement();
        
        //Allocation new memory with bitmaps
        void* allocateMem(unsigned int sizeAllocate);
        //free memory with bitmaps
        void freeMem(void* addressIn);
        
}

/* -- End -- Define data structure for memory management with bitmaps*/

#endif

