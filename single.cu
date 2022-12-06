#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <argp.h>


static char doc[] = "Single GPU version of the MDP solver";
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

#define index(i, j, k)  ((i)*(S)*(A)) + ((j)*(S)) + (k)


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
  float * StateT;
  float * StateR;
  float * BlockMaxs;
  float * StateMax;
  float * StateMaxCPU;
  float * next;
  float * V;
  float * T;
  float * R;
  int Sint;
  int Aint;

  //Sequential stuff
  float sum_seq;
  float max_a;
  float * V_seq;
  float * next_seq;

  StateMaxCPU = (float *)calloc(1, sizeof(float));
  StateMaxCPU[0] = 0;

  /**** Fill T and R ******/
  FullT = (float *)calloc(S*A*S, sizeof(float));
  FullR = (float *)calloc(S*A*S, sizeof(float));

  for (int tr = 0; tr<S*A*S;tr++){
    FullT[tr] = 1;
    FullR[tr] = 1;
  }


  /**** Fill V and V_seq ******/
  next = (float *)calloc(S, sizeof(float));
  V_seq = (float *)calloc(S, sizeof(float));
  for (int j = 0; j<S;j++){
    next[j] = 0;
    V_seq[j] = 0;
  }

  StateT = (float *)calloc(S*A, sizeof(float));
  StateR = (float *)calloc(S*A, sizeof(float));

  clock_t start = clock();

  /*** Allocate Required Space on GPU ***/
  cudaMalloc((void **)&V, S*sizeof(float));
  if(!V) {
	  printf("Error allocating array V\n");
	  exit(1);
	}
  cudaMalloc((void **)&T, S*A*sizeof(float));
  if(!T) {
	  printf("Error allocating array T\n");
	  exit(1);
	}
  cudaMalloc((void **)&R, S*A*sizeof(float));
  if(!R) {
	  printf("Error allocating array R\n");
	  exit(1);
	}
  cudaMalloc((void **)&BlockMaxs, ceil(A/threadperblock)*sizeof(float));
  if(!BlockMaxs) {
	  printf("Error allocating array BlockMaxs\n");
	  exit(1);
	}
  cudaMalloc((void **)&StateMax, 1*sizeof(float));
  if(!StateMax) {
	  printf("Error allocating array StateMax\n");
	  exit(1);
	}

  /**** Loop Through Iterations ******/
  for (int i = 0; i < numiters; i++) {
    // printf("i = %u\n", i);
    /*** Move next to V ***/
    cudaMemcpy(V, next, S*sizeof(float), cudaMemcpyHostToDevice);

    /*** Loop Through States per GPU ***/
    for (int k = 0; k < S; k++) {

      /*** Find the part of the full T and R arrays that are needed for this state. Fill new vectors ***/
      for (int j = 0; j < S*A; j++) {
        StateT[j] = FullT[k*Sint*Aint + j];
        StateR[j] = FullR[k*S*A + j];
      }

      /*** Move all required arrays to GPU ***/
      cudaMemcpy(T, StateT, S*A*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(R, StateR, S*A*sizeof(float), cudaMemcpyHostToDevice);

      /*** Run the Max and Sum for each block at a time. The number of total thread = number of actions.  ***/
      MaxSum<<<ceil(A/threadperblock), threadperblock>>>(BlockMaxs,V,R,T,k,A,S);
      cudaDeviceSynchronize();

      /*** Use second kernel to find max of all blocks.  ***/
      SecondReduc<<<1, ceil(A/threadperblock)/2>>>(StateMax,BlockMaxs);
      cudaDeviceSynchronize();

      /*** Save the state max. ***/
      cudaMemcpy(StateMaxCPU, StateMax, 1*sizeof(float), cudaMemcpyDeviceToHost);
      next[k] = StateMaxCPU[0];
    }
    // Synchronize all the GPUs before beginning next iteration.
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
      V_seq[s] = next_seq[s];
    }
  }

  for (int s = 0; s < S; s++) {
    if (V_seq[s] != next[s]) {
      printf("GPU version failed correctness test\n");
      exit(1);
    }
  }
  
  printf("GPU version passed correctness test\n");

  cudaFree(V);
  cudaFree(T);
  cudaFree(R);
  cudaFree(BlockMaxs);
  cudaFree(StateMax);
  free(StateMaxCPU);
  free(FullT);
  free(FullR);
  free(next);
  free(V_seq);
  free(StateT);
  free(StateR);
  free(next_seq);

  return 0;
}


__global__ void MaxSum(float *BlockMaxs,float * V, float * R,float * T,int sID,int A,int S)
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
      sprimeSum += T[aID*S + spID]*(R[aID*S + spID] + V[spID]);
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
      BlockMaxs[blockIdx.x] = sprimeSumValues[0];
    }
 }

 __global__ void SecondReduc(float * StateMax, float *BlockMaxs)
 {
     __syncthreads();
     // Use Reduction Tree to quickly find max of all blocks
     for (int stride = blockDim.x; stride >= 1; stride /= 2)
     {
       __syncthreads();
       if (threadIdx.x < stride) {
        BlockMaxs[threadIdx.x] = max(BlockMaxs[threadIdx.x],BlockMaxs[threadIdx.x + stride]);
     }
     }
     // Save the state max
     if (threadIdx.x == 0 ) {
      StateMax[0] = BlockMaxs[0];
    }
  }


// compile with: nvcc -o singlegpu single.cu
// ./singlegpu -A 1024 -S 100 -i 50
