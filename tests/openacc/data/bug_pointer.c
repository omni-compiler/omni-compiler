#include <stdio.h>
#define N 100
#define M 200

int main(void)
{
    int a[N];
    int b[M][N];
    int *a_p = a;
    int (*b_p)[N] = (int (*)[N])b;
    int i,j;
    for(i = 0; i < N; i++){
	a[i] = 0;
    }
    for(i = 0; i < M; i++){
	for(j = 0; j < N; j++){
	    b[i][j]= 0;
	}
    }    

#pragma acc parallel copy(a_p[3]) //[3:1] is ok but [3] is bad
    a_p[3] = 1234;

    if(a[3] != 1234){
	return 1;
    }

#pragma acc parallel loop copy(b_p[4][:])
    for(j = 0; j < N; j++){
	b_p[4][j] = j * 11;
    }

    for(j = 0; j < N; j++){
	if(b[4][j] != j * 11){
	    return 2;
	}
    }

    printf("PASS\n");
    return 0;
}
