#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>
#include "cuda_runtime.h"
#define N 200000

int main(void)
{
    int *a = (int*)malloc(sizeof(int)*N);
    if(a == NULL){
	return 1;
    }
    cudaStream_t st;
    cudaError_t error = cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    if(error != cudaSuccess){
	return 1;
    }

    acc_set_cuda_stream(2, st);

    for(int i = 0; i < N; i++){
	a[i] = i;
    }

#pragma acc data copyout(a[0:N])
    {
	int *dev_a;

#pragma acc host_data use_device(a)
	dev_a = a;

	cudaMemcpyAsync(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice, st);

#pragma acc parallel loop async(2)
	for(int i = 0; i < N; i++){
	    a[i] += i;
	}

#pragma acc wait(2)
    }

    for(int i = 0; i < N; i++){
	if(a[i] != i*2) return 1;
    }

    printf("PASS\n");
    return 0;
}
    
