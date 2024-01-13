#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

/*
It is a simple example for parallel computing in CUDA
*/


// CUDA kernel, one array add oneself one in each element
__global__ void addArrays(int* a, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        a[tid] = a[tid] + 1;
    }
}

int main()
{
    const int arraySize = 1024; // Size of the arrays
    const int blockSize = 256;  // Number of threads per block
    // Allocate memory on the host
    int* hostA = new int[arraySize];

    // Initialize arrays with  values
    int i = 0;
    for(i = 0; i < arraySize; i++)
    {
        hostA[i] = i;
    }

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) 
    {
    std::cerr << "No CUDA-compatible GPU device found." << std::endl;
    return 1;
    }

    // Allocate memory on the device (GPU)
    int deviceId = 0;
    cudaSetDevice(deviceId);
    int* deviceA;
    cudaMalloc((void**)&deviceA, arraySize * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(deviceA, hostA, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    int numBlocks = (arraySize + blockSize - 1) / blockSize;
    addArrays<<<numBlocks, blockSize>>>(deviceA, arraySize);

    // Copy the result back from the device to the host
    cudaMemcpy(hostA, deviceA, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(deviceA);

    // Display the results
    for (int i = 0; i < arraySize; ++i) {
        std::cout << hostA[i] << std::endl;
    }

    // Clean up host memory
    delete[] hostA;

    return 0;
}

