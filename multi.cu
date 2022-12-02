#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

__global__ void MaxSum(float *,float *,float *,float *,int,int,int);
__global__ void SecondReduc(float *,float *);

int main(int argc, char * argv[])
{

  if (argc != 4)
  {
    printf("Not enough arguments");
    exit(1);
  }

  int threadperblock = 64;
  float * FullT;
  float * FullR;
  float * BlockMaxArrays;
  float * V;
  int numDevs= 0;

  unsigned int S;
  unsigned int A;
  int numiters = 0;

  numiters = atoi(argv[3]);
  A =  (unsigned int) atoi(argv[1]);
  S =  (unsigned int) atoi(argv[2]);

  clock_t start = clock();

  /**** Fill T and R ******/
  // Allocate Unified Memory
  cudaMallocManaged(&FullT, S*A*S*sizeof(float));
  cudaMallocManaged(&FullR, S*A*S*sizeof(float));
  cudaMallocManaged(&V, S*sizeof(float));
  cudaMallocManaged(&BlockMaxArrays, S*ceil(A/threadperblock)*sizeof(float));

  for (int tr = 0; tr<S*A*S;tr++){
    FullT[tr] = 1;
    FullR[tr] = 1;
  }

  /**** Fill next ******/
  for (int j = 0; j<S;j++){
    V[j] = 0;
  }

  // Find number of GPUs
  cudaGetDeviceCount(&numDevs);

  /**** Loop Through Iterations ******/
  for (int i = 0; i < numiters; i++) {
    printf("i = %u\n", i);

  /*** Loop Through States per GPU ***/
  for (int k = 0; k < S/numDevs; k++) {

  /*** Loop Through GPU Devices ***/
  for (int d = 0; d < numDevs; d++) {

    cudaSetDevice(d);

    /*** Run the Max and Sum for each block at a time. The number of total thread = number of actions.  ***/
    MaxSum<<<ceil(A/threadperblock), threadperblock>>>(BlockMaxArrays,V,FullR,FullT,k*numDevs+d,A,S);
  }
  }

  for (int d = 0; d < numDevs; d++) {
    cudaSetDevice(d);
    cudaDeviceSynchronize();
  }

  /*** Use second kernel to find max of all blocks.  ***/
  SecondReduc<<< S , ceil(A/threadperblock)/2>>>(V,BlockMaxArrays);
  }

  for (int d = 0; d < numDevs; d++) {
    cudaSetDevice(d);
    cudaDeviceSynchronize();
  }

  clock_t end = clock();
  double time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  printf("Time taken = %lf\n", time_taken);

  next_seq = (float *)calloc(S, sizeof(float));
  /*** Do sequential. ***/
  for (int i = 0; i < numiters; i++) {
    // printf("i = %u\n", i);
    for (int s = 0; s < S; s++) {
      max_a = 0;
      for (int a = 0; a < A; a++) {
        sum_seq = 0;
        for (int sp = 0; sp < S; sp++) {
          sum_seq = sum_seq + FullT[s*S*A + a*S + sp]*(FullR[s*S*A + a*S + sp] + V_seq[sp]);
        }
        if (sum_seq > max_a){
          max_a = sum_seq;
        }
      }
      next_seq[s] = max_a;
    }
    for (int s = 0; s < S; s++) {
      if (V_seq[s] != next[s]){
       printf("FAIL\n");
     }
    }
  }

}

__global__  void MaxSum(float *BlockMaxArrays,float * V, float * FullR,float * FullT,int sID,int A,int S)
{
    // Use Shared Memory to write sums
    __shared__ float sprimeSumValues[64];
    float sprimeSum;
    int aID;
    sprimeSum = 0;
    // Action ID
    aID = blockIdx.x*blockDim.x + threadIdx.x;

    // Loop all s' and perform sum
    for (int spID = 0; spID < S; spID ++)
    {
      sprimeSum += FullT[S*A*sID + aID*S + spID]*(FullR[S*A*sID + aID*S + spID] + V[spID]);
    }
    // Save s prime sum value
    sprimeSumValues[threadIdx.x] = sprimeSum;

    //Wait till all threads have done this
    __syncthreads();
    // Use Reduction Tree to quickly find max of each block
    for ( int stride = blockDim.x/2; stride >= 1; stride /= 2)
    {
      __syncthreads();
      if (threadIdx.x < stride) {            // aID is thread (action) ID
       sprimeSumValues[threadIdx.x] = max(sprimeSumValues[threadIdx.x],sprimeSumValues[threadIdx.x + stride]);
    }
    }
    if (threadIdx.x == 0 ){
      // Save the block max and then find total max somehow
      BlockMaxArrays[sID*gridDim.x + blockIdx.x] = sprimeSumValues[0];
    }
 }

 __global__  void SecondReduc(float * V, float *BlockMaxArrays)
 {
     __syncthreads();
     // Use Reduction Tree to quickly find max of all blocks
     for (int stride = blockDim.x; stride >= 1; stride /= 2)
     {
       __syncthreads();
       if (threadIdx.x < stride) {
         BlockMaxArrays[blockIdx.x*blockDim.x*2 + threadIdx.x] = max(BlockMaxArrays[blockIdx.x*blockDim.x*2 + threadIdx.x],BlockMaxArrays[blockIdx.x*blockDim.x*2 + threadIdx.x + stride]);
     }
     }
     // Save the state max
     if (threadIdx.x == 0) {
      V[blockIdx.x] = BlockMaxArrays[blockIdx.x*blockDim.x*2];
    }
  }
