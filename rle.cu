#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>

#include <sys/time.h>


#include "cuda.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define N 1000
#define NBLOCKS 1
#define BLOCK_SIZE 1024

// a function that calculates the time difference in msec
unsigned int TimeDiff(struct timespec start, struct timespec end)
{
    return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
}


unsigned int FillWithRandomNumbers(int* values, unsigned char* lengths, int n)
{
    unsigned int totalLength=0;

    srand((unsigned)time(0));
    for (int i=0;i<N;i++)
    {
        *values=rand() % 0xffffffff;
        unsigned char length=rand() & 0xff;
        *lengths = length;
        totalLength+=(unsigned int)length;
        values++;
        lengths++;
    }

    return totalLength; // the decoded mesage will need this much of space.
}

__global__ void decodeRLE(int* values, unsigned char * lengths, unsigned int* np, unsigned int * output)
{
    unsigned int n=*np;
    extern __shared__ unsigned int sumOfLengths[];

    cg::grid_group grid=cg::this_grid();
    int threadId=blockIdx.x * blockDim.x + threadIdx.x;

    /// we have only half of the threads. each take care of the odd and even.
    int evenIndex=threadId*2;

    // step 0
    sumOfLengths[threadId]=(threadId==0 ? 0 : lengths[evenIndex-1])+lengths[evenIndex]; 
    grid.sync();

    // steps 1 .. n-1
    for (unsigned int groupSize=1, twiceGroupSize=2;twiceGroupSize<n;groupSize=twiceGroupSize, twiceGroupSize*=2)
    {
        unsigned int groupNr=threadId/twiceGroupSize;
        unsigned int minThreadId=groupSize*(1+groupNr*2);
        unsigned int maxThreadId=minThreadId+groupSize-1;
        if (threadId>=minThreadId && threadId<=maxThreadId)
        {
            unsigned int toAdd=sumOfLengths[minThreadId-1]; 
            sumOfLengths[threadId]+=toAdd;
        }
        grid.sync();
    }

    // last step. Fill even.
    int bothOffsets[2];
    bothOffsets[0]=threadId==0 ? 0 : sumOfLengths[threadId-1]+lengths[evenIndex-1];
    bothOffsets[1]=sumOfLengths[threadId];

    grid.sync();

    // now do real decompression and place everything in output buffer. Do for both indexes
    for (int index=0;index<=1;index++) // just 2 passes
    {
        for (int i=bothOffsets[index];i<bothOffsets[index]+(unsigned int)lengths[evenIndex+index];i++)
        {
            output[i]=values[evenIndex+index];
        }    
    }
}

#define NLOOPS 20


int main()
{
    int *values;
    unsigned int *n;
    unsigned char *lengths;

    cudaMallocManaged(&n,sizeof(unsigned int)); // n contains the size of the RLE sequence
    cudaMallocManaged(&values,N*sizeof(int));
    cudaMallocManaged(&lengths,N*sizeof(unsigned char));

    *n=N;
    // create a long rle sequence
    unsigned int totalLength=FillWithRandomNumbers(values, lengths, N);

    printf("total length of exploded RLE: %ld\n", totalLength);

    // the place where the decompressed message should end up
    unsigned int* output;
    cudaMallocManaged(&output,totalLength*sizeof(unsigned int));

    cudaDeviceSynchronize();

    void *kernelArgs[] = 
    {
        (void*) &values, (void *)&lengths, (void*)&n, (void *) &output
    };

    // start of the real work
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    for (int count=0;count<NLOOPS;count++)
    {

        cudaLaunchCooperativeKernel((void*)decodeRLE, // function
                                        1, // grid dimensions
                                        N/2, // block dimensions
                                        kernelArgs, 
                                        N/2*sizeof(unsigned int) // shared memory. Unsigned ints because summing multiple unsigned chars > 255
                                    );
        cudaDeviceSynchronize();   
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    unsigned int timeElapsed = TimeDiff(start, end); // in microseconds
    unsigned int timePerRLE = timeElapsed / NLOOPS;
    printf("time per RLE %ld microsecs\n", timePerRLE);

}

