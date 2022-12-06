#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <argp.h>

static char doc[] = "Multi GPU version of the MDP solver";
static char args_doc[] = "";

static struct argp_option options[] = {
  {0, 'A',  "A_size",  0,  "Action space size" },
  {0, 'S',  "S_size",	 0,  "State space size" },
  {0, 'i',  "iter",    0,  "Number of iterations" },
  { 0 }
};

struct arguments {
  char* args[0];
  int A_size, S_size, iter;
};

static error_t parse_opt (int key, char* arg, struct argp_state* state) {
  struct arguments *arguments =  static_cast<struct arguments*>(state->input);

  switch (key) {
    case 'A':
      arguments->A_size = atoi(arg);
      break;
    case 'S':
      arguments->S_size = atoi(arg);
      break;
    case 'i':
      arguments->iter = atoi(arg);
      break;
    default:
      return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp argp = { options, parse_opt, args_doc, doc };


__global__ void MaxSum(float *,float *,float *,float *,int,int,int);
__global__ void SecondReduc(float *,float *);

int main(int argc, char * argv[])
{
  struct arguments user_input;

  /* Default values. */
  user_input.A_size = 1024;
  user_input.S_size = 10;
  user_input.iter = 1;

  argp_parse (&argp, argc, argv, 0, 0, &user_input);
  printf ("Action space size = %d\nState space size = %d\nNumber of iterations = %d\n",
    user_input.A_size, user_input.S_size, user_input.iter);

  unsigned int A =  (unsigned int) user_input.A_size;
  unsigned int S =  (unsigned int) user_input.S_size;
  int numiters = user_input.iter;

  int threadperblock = 64;
  float * FullT;
  float * FullR;
  float * BlockMaxArrays;
  float * V;
  int numDevs= 0;

  //Sequential stuff
  float sum_seq;
  float max_a;
  float * V_seq;
  float * next_seq;

  next_seq = (float *)calloc(S, sizeof(float));
  V_seq = (float *)calloc(S, sizeof(float));
  for (int j = 0; j<S;j++){
    next_seq[j] = 0;
    V_seq[j] = 0;
  }
  clock_t start = clock();

  /**** Fill T and R ******/
  // Allocate Unified Memory
  cudaMallocManaged(&FullT, S*A*S*sizeof(float));
  if(!FullT) {
	  printf("Error allocating array FullT\n");
	  exit(1);
	}
  cudaMallocManaged(&FullR, S*A*S*sizeof(float));
  if(!FullR) {
	  printf("Error allocating array FullR\n");
	  exit(1);
	}
  cudaMallocManaged(&V, S*sizeof(float));
  if(!V) {
	  printf("Error allocating array V\n");
	  exit(1);
	}
  cudaMallocManaged(&BlockMaxArrays, S*ceil(A/threadperblock)*sizeof(float));
  if(!BlockMaxArrays) {
	  printf("Error allocating array BlockMaxArrays\n");
	  exit(1);
	}

  for (int tr = 0; tr<S*A*S;tr++){
    FullT[tr] = tr/1000;
    FullR[tr] = tr/1000;
  }

  for (int j = 0; j<S;j++){
    V[j] = 0;
  }

  // Find number of GPUs
  cudaGetDeviceCount(&numDevs);
  /**** Loop Through Iterations ******/
  for (int i = 0; i < numiters; i++) {

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


  for (int d = 0; d < numDevs; d++) {
    cudaSetDevice(d);
    cudaDeviceSynchronize();
  }
}

  clock_t end = clock();
  double time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  printf("Time taken = %lf\n", time_taken);

  next_seq = (float *)calloc(S, sizeof(float));
  /*** Do sequential. ***/
  for (int i = 0; i < numiters; i++) {
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
      V_seq[s] = next_seq[s];
    }
  }

  for (int s = 0; s < S; s++) {
    if (V_seq[s] != V[s]) {
      printf("GPU version failed correctness test\n");
      exit(1);
    }
  }
  printf("GPU version passed correctness test\n");

  cudaFree(FullT);
  cudaFree(FullR);
  cudaFree(V);
  cudaFree(BlockMaxArrays);
  free(V_seq);
  free(next_seq);

  return 0;
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


// compile with: nvcc -o multigpu multi.cu
// ./multigpu -A 1024 -S 100 -i 50
