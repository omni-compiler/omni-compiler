#include <stdio.h>
#include <xmp.h>

#define N 16
#define NP 2
#pragma xmp nodes p[NP]
#pragma xmp template t[N]
#pragma xmp distribute t[block] onto p

int main()
{
    int a[N][N];
#pragma xmp align a[i][*] with t[i]
#pragma xmp shadow a[1][0]

    int i,j;
#pragma xmp loop (i) on t[i]
    for(i = 0; i < N; i++){
	for(j = 0; j < N; j++){
	    a[i][j] = 100 * i + j;
	}
    }

#pragma acc data copy (a)
    {
#pragma xmp reflect (a) acc
    }

    int rank  = xmp_node_num() - 1;
    int begin = N/NP * rank;
    int end   = N/NP * (rank+1);

    if(rank != 0) begin--;
    if(rank != NP-1) end--;

    int err = 0;

    for(i = begin; i < end; i++){
	for(j = 0; j < N; j++){
	    if(a[i][j] != 100 * i + j) err = 1;
	}
    }

#pragma xmp reduction(+:err)

#pragma xmp task on p[0]
    if(err == 0){
	printf("PASS\n");
    }else{
	return 1;
    }

    return 0;
}
