# SyncMalloc
SyncMalloc is the first repository to support GPU dynamic allocation across all scales co-managed with the Host.

If this framework contributes to your research or projects, kindly acknowledge it by citing the relevant papers that align with the functionalities you utilize.

Jiajian Zhang, Fangyu Wu, Hai Jiang, Guangliang Cheng, Genlang Chen, and Qiufeng Wang. 2024. SyncMalloc: A Synchronized Host-Device Co-Management System for GPU Dynamic Memory Allocation across All Scales. In The 53rd International Conference on Parallel Processing (ICPP ’24), August 12–15, 2024, Gotland, Sweden. ACM, New York, NY, USA, 10 pages.

DOI Link:  https://doi.org/10.1145/3673038.3673069

# How to use?
SyncMalloc is thoroughly evaluated through diverse performance tests detailed in the directory *./Experiments* or *.src/MMUOnCPUPingPongFinalVersion*. These evaluations include the **Singular Mixed-size Test**, **Scaling Test**, **Random Allocation Test**, and **detailed Breakdowns of the system**.

## Singular Mixed-size Test and Scaling Test
To demonstrate the performance of both the Singular Mixed-size Test and the Scaling Test, please refer to the *./Experiments/singleAllocation/MMUOnCPUPingPongFinalVersion/singleAllocation.cu* example file. This file utilizes the *HMCoAllocate* function to dynamically allocate memory per thread. The size of the memory allocation can be specified within each thread. The number of threads can be configured using *BLOCKNUMBER* and *BLOCKSIZE* in the *MMUOnCPU.hpp* file.

To complie:
`nvcc -rdc=true singleAllocation.cu MMUOnCPU.cu -lpthread`

## Random Allocation Test
The Random Allocation Test evaluates the efficiency of SyncMalloc under rigorous conditions using the files *./Experiments/randomAllocation/randomAllocationLarge.cu* for large allocations and  *testshbenchOnFirstClass.cu* for smaller allocations. These tests involve multiple threads performing a series of random-sized memory operations including allocations and deallocations. Each thread runs for *n* times, specified by *ALL_FREE_TIMES* in the code, and can vary from 0.5K to 10K iterations. The operations, whether allocations or deallocations, are randomly generated, stored in the *d_isAllocate* and *d_freeBlock* arrays, and executed by the GPU based on these instructions.

To complie:
`nvcc -rdc=true randomAllocationLarge.cu MMUOnCPU.cu -lpthread`

or 

`nvcc -rdc=true testshbenchOnFirstClass.cu MMUOnCPU.cu -lpthread`

Note: For tests involving small allocations, set *BLOCKSIZE* to 16, because the benchmark of the Random Allocation Test is designed to initialize a small array to store the unique scope. But, the *BLOCKNUMBER* can be adjusted as needed to accommodate the unique scope of the Random Allocation Test.

## Eetailed Breakdowns of the system
The detailed performance breakdowns are conducted using directories *./Experiments/selfEvalutation/MMUOnCPUPingPongV1.3.6BrekDownsGPU* and *./Experiments/selfEvalutation/MMUOnCPUPingPongV1.3.6BrekDownsHosts*. These tests analyze the overhead by inserting time measurement functions in the function, *blockAllocationThr*, where each thread deals with block allocations on the host. This setup provides insight into the efficiency and scalability of system operations across both GPU and Host environments.







