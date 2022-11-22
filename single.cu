#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

__global__ void MaxSum();

#define index(i, j, k)  ((i)*(S)*(A)) + ((j)*(S)) + (k)

int main(float * T ,float * R)
{
  int numDevs= 0;

  unsigned int S;
  unsigned int A;

  next = (float *)calloc(S, sizeof(float));

  /*** Allocate Required Space on GPU ***/
  cudaMalloc((void **)&V, S*sizeof(float));
  cudaMalloc((void **)&T, S*A*sizeof(float));
  cudaMalloc((void **)&R, S*A*sizeof(float));
  cudaMalloc((void **)&BlockMaxs, ceil(A/1024)*sizeof(float));

  /**** Loop Through Iterations ******/
  for (int i = 0;i < numiters; i++) {

  /*** Move next to V ***/
  cudaMemcpy(V, next, S*sizeof(float), cudaMemcpyHostToDevice);

  /*** Loop Through States per GPU ***/
  for (int k = 0; k < S; k++) {

    /*** Find the part of the full T and R arrays that are needed for this state ***/
    StateT
    StateR

    /*** Move all required arrays to GPU ***/
    cudaMemcpy(T, StateT, S*A*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(R, StateR, S*A*sizeof(float), cudaMemcpyHostToDevice);

    /*** Run the Max and Sum for each block at a time. The number of total thread = number of actions.  ***/
    MaxSum<<<ceil(A/1024), 1024>>>(args);

    cudaDeviceSynchronize();
    cudaFree(T);
    cudaFree(R);

    /*** Use second kernel to find max of all blocks.  ***/
    SecondReduc<<<1, ceil(A/1024)>>>(args);

    /*** Save the state max. ***/
    cudaMemcpy(next[k], StateMax, int, cudaMemcpyDeviceToHost);

  }
  // Synchronize all the GPUs before beginning next iteration.
  cudaDeviceSynchronize();
  }

}

__global__  void MaxSum(int sID,)
{
    // Use Shared Memory to write sums
    __shared__ float sprimeSumValues[A];

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
    BlockMaxs[blockIdx.x] = sprimeSumValues[0];

 }

 __global__  void SecondReduc(int StateMax, float * threadmax)
 {
     int StateMax;
     // Use Reduction Tree to quickly find max of all blocks
     for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2)
     {
       __syncthreads();
       if (threadIdx.x < stride) {
        BlockMaxs[threadIdx.x] = max(BlockMaxs[threadIdx.x],BlockMaxs[threadIdx.x + stride]);
     }
     }
     // Save the state max
     StateMax = BlockMaxs[0];

  }
