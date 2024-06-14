#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "LinkedListMemory.hpp"

int main() 
{
    MemoryManagement memoryManagement;
    // int* a = (int*)memoryManagement.allocateMem(sizeof(int)*1024*2);
    // int* b = (int*)memoryManagement.allocateMem(sizeof(int));
    // int* c = (int*)memoryManagement.allocateMem(sizeof(int)*1024*4);
    // memoryManagement.freeMem(a);
    // memoryManagement.freeMem(b);
    // memoryManagement.freeMem(c);
    // int* d = (int*)memoryManagement.allocateMem(sizeof(int)*1024*4);
    // memoryManagement.freeMem(d);
    int* e = (int*)memoryManagement.allocateMem(sizeof(int)*1024*1024*8);
    
    int* f = (int*)memoryManagement.allocateMem(sizeof(int)*1024*1024*10);
    int* g = (int*)memoryManagement.allocateMem(sizeof(int)*1024*1024*16);
    memoryManagement.freeMem(f);
    int* h = (int*)memoryManagement.allocateMem(sizeof(int)*1024*1024*16);
    int* i = (int*)memoryManagement.allocateMem(sizeof(int)*1024*1024*9);

    
    memoryManagement.freeMem(g);
    
    memoryManagement.freeMem(h);
    
    memoryManagement.freeMem(e);
    memoryManagement.freeMem(i);

}