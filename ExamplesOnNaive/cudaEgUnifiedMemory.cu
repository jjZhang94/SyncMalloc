#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/resource.h>
#include <iostream>

#define NUMBERPERTHREAD 16  //how many numbers each thread is to deal with.

/*
This is an example to illustarte an overscubscription in GPU via unified memory.
*/

// CUDA kernel, one array add oneself one in each element
__global__ void addArrays(int* a, long size) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    tid = tid * NUMBERPERTHREAD;
    if (tid < size) {
        int i = 0;
        for(i=0; i<NUMBERPERTHREAD; i++)
        {
            a[tid+i] = a[tid+i] * 2;
        }
    }
}


int main(){
    long N = 4000000000; // Large size, e.g., 100 million elements
    //long N = 80000000;
    size_t size = N * sizeof(int) * NUMBERPERTHREAD;
    printf("Size = %ld", size);
    int *a;

    //memory usage structure
    struct rusage usage;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&a, size);

    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        long rss = usage.ru_maxrss;

        // Convert kilobytes to gigabytes
        double gb_usage = rss / (1024.0 * 1024.0);

        printf("Memory usage: %.2f GB\n", gb_usage);
    } else {
        printf("Failed to retrieve memory usage.\n");
    }

    // Initialize vectors on the host
    long i;
    for (i = 0; i < (N*NUMBERPERTHREAD); i++) {
        a[i] = 2;
    }
    

    // Kernel launch parameters
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Execute the kernel
    printf("kernel is ready to start\n");
    addArrays<<<numBlocks, blockSize>>>(a, N);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    printf("kernel finished\n");

    // Check for errors (all values should be 2.0f)
    bool error = false;
    for (i = 0; i < N; i++) {
        if (a[i] != 4) {
            printf("Error in index %ld\n", i);
            printf("a in i = %d\n", a[i]);
            error = true;

            //print others 
            for (int j = 0; j < 1000; j++)
            {
                printf("%d ", a[j+1]);
            }
            break;
        }
    }

    if (error) {
        std::cout << "Error: Incorrect results" << std::endl;
    } else {
        std::cout << "Success: All results are correct" << std::endl;
    }

    // Free memory
    cudaFree(a);

    return 0;
}
