int bitmap[1024*1024];

void clearBits(unsigned int size, unsigned int startBlockIndex)
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