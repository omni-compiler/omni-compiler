#define N 128
#include <stdio.h>
#include <xmp.h>

#pragma xmp nodes p(*)
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(block) onto p

int main(void)
{
    int i, sum, sum2;

    sum = 0;
    sum2 = 0;

#pragma acc data copy(sum) copyin(sum2)
    {
#pragma xmp loop (i) on t(i)
#pragma acc parallel loop reduction(+:sum, sum2)
	for(i = 0; i < N; i++){
	    sum  += (i + 1);
	    sum2 += (i + 1);
	}

#pragma xmp reduction(+:sum) acc
#pragma xmp reduction(+:sum2)
    }

    if(sum == N*(N+1)/2 && sum2 == 0){
#pragma xmp task on p(1)
	printf("OK\n");
    }else{
	printf("[%d] invalid result, sum=%d, sum2=%d\n", xmp_node_num(), sum, sum2);
	return 1;
    }

    return 0;
}
