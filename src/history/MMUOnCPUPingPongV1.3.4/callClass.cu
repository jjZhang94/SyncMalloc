
#include <cuda_runtime.h> // For CUDA runtime APIs (cudaMalloc, cudaMemcpy, etc.)
#include <iostream> 

class SimpleClass {
public:
    SimpleClass(int initialValue)
    {
        for(int i = 0; i<3; i++)
        {
            value[i] = 3; 
        }
        printf("a\n");
    }

    ~SimpleClass()
    {
        printf("b\n");
    }

    __device__ void increment() {
        value[0]++;
    }

    __device__ int getValue() const {
        return value[3];
    }

private:
    int value[3];
};

__global__ void useSimpleClass(SimpleClass* a) {
    int tid = threadIdx.x;
    a->increment();
    printf("value: %d\n", a -> getValue());
}

int main() {
    SimpleClass* d_obj;

    // Allocate memory for the object on the device
    cudaMalloc(&d_obj, sizeof(SimpleClass));
    

    // Create an instance of the class on the device
    // Note: cudaMemcpyFromSymbol or a kernel might be used to initialize device-side objects in real scenarios
    SimpleClass h_obj(0); // Create a host object for example purposes
    cudaMemcpy(d_obj, &h_obj, sizeof(SimpleClass), cudaMemcpyHostToDevice);

    // Launch the kernel
    useSimpleClass<<<1, 1>>>(d_obj);
    cudaDeviceSynchronize();

    // Copy the object back to host to check the result
    // cudaMemcpy(&h_obj, d_obj, sizeof(SimpleClass), cudaMemcpyDeviceToHost);
    // std::cout << "Value after increment: " << h_obj.getValue() << std::endl; // This will not compile; illustrated for conceptual understanding

    // Free the device memory
    // cudaFree(d_obj);

    return 0;
}
