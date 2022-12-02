#include <stdlib.h>
#include <stdio.h>
#include <time.h>


#define index(i, j, k)  ((i)*(S)*(A)) + ((j)*(S)) + (k)

int main(int argc, char * argv[])
{
    if (argc != 4)
    {
        printf("Not enough arguments");
        exit(1);
    }

    float * FullT;
    float * FullR;
    float * StateT;
    float * StateR;
    float * BlockMaxs;
    float * StateMax;
    float * next;
    float * V;
    float * T;
    float * R;

    float sum_seq;
    float max_a;
    float * V_seq;
    float * next_seq;

    unsigned int S;
    unsigned int A;
    int numiters = 0;

    numiters = atoi(argv[3]);
    A =  (unsigned int) atoi(argv[1]);
    S =  (unsigned int) atoi(argv[2]);

    /**** Fill T and R ******/
    FullT = (float *)calloc(S*A*S, sizeof(float));
    FullR = (float *)calloc(S*A*S, sizeof(float));

    for (int tr = 0; tr<S*A*S;tr++){
        FullT[tr] = 1;
        FullR[tr] = 1;
    }


    /**** Fill V and V_seq ******/
    V_seq = (float *)calloc(S, sizeof(float));
    for (int j = 0; j<S;j++){
        V_seq[j] = 0;
    }
    

    next_seq = (float *)calloc(S, sizeof(float));

    clock_t start = clock();

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

    clock_t end = clock();  
    double time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time taken = %lf\n", time_taken);

}

