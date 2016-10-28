#include <stdio.h>
#include <stdlib.h>

#define M 5
#define N 7

int **array_pp;
int array_dst[M][N];

static inline
void init_array()
{
    for(int i = 0; i < M; i++){
	for(int j=0;j<N;j++){
	    array_pp[i][j] = (i*3 + j*5);
	}
    }
}

static inline
int compare(const int i, const int j, const int result, const int valid)
{
    if(result != valid){
	printf("error at [%d][%d], expected %d but %d\n", i, j, valid, result);
	return 1;
    }
    return 0;
}

int main()
{
    int i,j,k;

    //alloc array_pp
    array_pp = (int **)malloc(M * sizeof(int*));
    for(i = 0; i < M; i++){
	array_pp[i] = (int *)malloc(N * sizeof(int));
    }


/*----------------------------------------------------------------------*/
/* create [0:M][0:N] & update all                                       */
/*----------------------------------------------------------------------*/
    init_array();
#pragma acc data create(array_pp[0:M][0:N])
    {
#pragma acc update device(array_pp)
#pragma acc parallel loop copy(array_dst)
	for(int i = 0; i < M; i++){
	    for(int j = 0; j < N; j++){
		array_dst[i][j] = array_pp[i][j];
	    }
	}
	//check
	for(i = 0; i < M; i++)
	    for(j = 0; j < N; j++)
		if(compare(i, j, array_dst[i][j], (i*3+j*5))) return 1;

    }


/*----------------------------------------------------------------------*/
/* copy [1:M-2][0:N]                                                    */
/*----------------------------------------------------------------------*/
    init_array();
#pragma acc parallel loop copy(array_pp[1:M-2][0:N])
    for(int i = 1; i < M-1; i++){
	for(int j = 0; j < N; j++){
	    array_pp[i][j]++;
	}
    }
    //check
    for(i = 0; i < M; i++){
	for(j = 0; j < N; j++){
	    if(i >= 1 && i <= M-2){
		if(compare(i, j, array_pp[i][j], (i*3+j*5+1))) return 2;
	    }else{
		if(compare(i, j, array_pp[i][j], (i*3+j*5))) return 2;
	    }
	}
    }


/*----------------------------------------------------------------------*/
/* copy [1:M-2][1:N-2]                                                  */
/*----------------------------------------------------------------------*/
    init_array();
#pragma acc parallel loop copy(array_pp[1:M-2][1:N-2])
    for(int i = 1; i < M-1; i++){
	for(int j = 1; j < N-1; j++){
	    array_pp[i][j]++;
	}
    }
    //check
    for(i = 0; i < M; i++){
	for(j = 0; j < N; j++){
	    if(i >= 1 && i <= M-2 && j >= 1 && j <= N-2){
		if(compare(i, j, array_pp[i][j], (i*3+j*5+1))) return 3;
	    }else{
		if(compare(i, j, array_pp[i][j], (i*3+j*5))) return 3;
	    }
	}
    }


/*----------------------------------------------------------------------*/
/* create [0:M][0:N] & update [1:M-2][1:N-2]                            */
/*----------------------------------------------------------------------*/
    init_array();
#pragma acc data copyin(array_pp[0:M][0:N])
    {
#pragma acc parallel loop
        for(int i = 0; i < M; i++){
	    for(int j = 0; j < N; j++){
		array_pp[i][j]++;
	    }
	}
#pragma acc update host(array_pp[1:M-2][1:N-2])
    }
    //check
    for(i = 0; i < M; i++){
	for(j = 0; j < N; j++){
#ifdef _OPENACC
	    if(i >= 1 && i <= M-2 && j >= 1 && j <= N-2){
		if(compare(i, j, array_pp[i][j], (i*3+j*5+1))) return 4;
	    }else{
		if(compare(i, j, array_pp[i][j], (i*3+j*5))) return 4;
	    }
#else
	    if(compare(i, j, array_pp[i][j], (i*3+j*5+1))) return 4;
#endif
	}
    }


/*----------------------------------------------------------------------*/
/* copyin [1:M-2][1:N-2] & update [1:M-2][1:N-2]                        */
/*----------------------------------------------------------------------*/
    init_array();
#pragma acc data copyin(array_pp[1:M-2][1:N-2])
    {
#pragma acc parallel loop
	for(int i = 1; i < M-1; i++){
	    for(int j = 1; j < N-1; j++){
		array_pp[i][j]++;
	    }
	}
#pragma acc update host(array_pp[1:M-2][1:N-2])
    }
    //check
    for(i = 0; i < M; i++){
	for(j = 0; j < N; j++){
	    if(i >= 1 && i <= M-2 && j >= 1 && j <= N-2){
		if(compare(i, j, array_pp[i][j], (i*3+j*5+1))) return 5;
	    }else{
		if(compare(i, j, array_pp[i][j], (i*3+j*5))) return 5;
	    }
	}
    }

    printf("PASS\n");
    return 0;
}
