#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include "LinkedListMemory.h"


unsigned char bitmap[4];
void setBitMapTrue(unsigned int blockIndex);
void setBitsMapTrue(size_t size, int startBolckIndex);

void setBitMapTrue(unsigned int blockIndex)
{   
    unsigned int byteIndex = blockIndex / 8;
    unsigned int bitIndex = blockIndex % 8;
    unsigned char mask = 1 << bitIndex;
    bitmap[byteIndex] |= mask;
}

void setBitsMapTrue(size_t size, int startBolckIndex)
{
    unsigned int endBlockIndex = startBolckIndex + size;
    unsigned int startByteIndex = startBolckIndex / 8;
    unsigned int endByteIndex = (endBlockIndex - 1) / 8;

    if (startByteIndex == endByteIndex)
    {
        // If size is no larger than 8 and within a single byte
        int startBitIndex = startBolckIndex % 8;
        int endBitIndex = endBlockIndex % 8;
        unsigned char mask = ((0xFF >> (8 - endBitIndex)) & (0xFF << startBitIndex));

        bitmap[startByteIndex] |= mask;
        return;
    }

    for (unsigned int i = startByteIndex; i <= endByteIndex; i++) {
        unsigned char mask;

        if (i == startByteIndex) {
            // Create mask for the first byte
            int startBitIndex = startBolckIndex % 8;
            mask = (0xFF << startBitIndex);
        } else if (i == endByteIndex) {
            // Create mask for the last byte
            int endBitIndex = endBlockIndex % 8;
            mask = (0xFF >> (8 - endBitIndex));
        } else {
            // Set all bits in the intermediate bytes
            mask = 0xFF;
        }

        bitmap[i] |= mask;
    }
}

int main() 
{
    memset(bitmap, 0, sizeof(bitmap));

    setBitsMapTrue(11,2);
    unsigned char c1 = bitmap[0];
    unsigned char c2 = bitmap[1];
    unsigned char c3 = bitmap[2];
    unsigned char c4 = bitmap[3];
    // char cc = 0;
    printf("%u, %u, %u, %u",c1, c2, c3, c4);
    return 0;
}