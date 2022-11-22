#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

__global__ void MaxSum();

#define index(i, j, k)  ((i)*(S)*(A)) + ((j)*(S)) + (k)

int main( )
{
  int numDevs= 0;

  unsigned int S;
  unsigned int A;

  // Allocate Unified Memory
  cudaMallocManaged(&T, S*A*S*sizeof(float));
  cudaMallocManaged(&R, S*A*S*sizeof(float));

  // Allocate Memory (Should this be unified?)
  cudaMallocManaged(&V, S*sizeof(float));
  cudaMallocManaged(&BlockMax, ceil(A/1024)*sizeof(float));

  // Find number of GPUs
  cudaGetDeviceCount(&numDevs);

  /**** Loop Through Iterations ******/
  for (int i = 0;i < numiters; i++) {

  /*** Loop Through States per GPU ***/
  for (int k = 0; k < S/numDevs; k++) {

  /*** Loop Through GPU Devices ***/
  for (int d = 0; d < numDevs; d++) {
    cudaSetDevice(d);

    /*** Run the Max and Sum for each block at a time. The number of total thread = number of actions.  ***/
    MaxSum<<<ceil(A/1024), 1024>>>(args);
    cudaDeviceSynchronize();

    /*** Use second kernel to find max of all blocks.  ***/
    SecondReduc<<<1, ceil(A/1024)>>>(args);

    /*** Save the state max. ***/
    cudaMemcpy(V[k*numDevs + d], StateMax, int, cudaMemcpyDeviceToHost);
  }
  }

  /*** Update the V matrix. ***/


  // Synchronize all the GPUs before beginning next iteration.

  cudaDeviceSynchronize();
  }

}

__global__  void MaxSum(int sID,)
{

    // Use Shared Memory to write sums
    __shared__ float sprimeSumValues[A];
    int BlockMax;

    // Action ID
    aID = blockIdx.x*blockDim.x + threadIDx.x;

    // Loop all s' and perform sum
    for (int spID = 0; spID < S; spID ++)
    {
      sprimeSum += T[index(sID,aID,spID)]*(R[index(sID,aID,spID)] + V[spID]);
    }

    // Save s prime sum value
    sprimeSumValues[threadId.x] = sprimeSum;

    //Wait till all threads have done this
    __syncthreads();

    // Use Reduction Tree to quickly find max of each block
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2)
    {
      __syncthreads();
      if (threadIdx.x < stride) {            // aID is thread (action) ID
       sprimeSumValues[threadIdx.x] = max(sprimeSumValues[threadIdx.x],sprimeSumValues[threadIdx.x + stride]);
    }
    }

    // Save the block max and then find total max somehow
    BlockMax[blockIdx.x] = sprimeSumValues[0];

 }

 __global__  void SecondReduc(int StateMax, float * threadmax)
 {

     int StateMax;

     // Use Reduction Tree to quickly find max of all blocks
     for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2)
     {
       __syncthreads();
       if (threadIdx.x < stride) {
        BlockMax[threadIdx.x] = max(BlockMax[threadIdx.x],BlockMax[threadIdx.x + stride]);
     }
     }
     // Save the state max
     StateMax = BlockMax[0];

  }
