#include <stdio.h>

int main(void)
{
    int i, sum;

    sum = 0;
#pragma acc parallel loop reduction(+:sum)
    for(i = 1; i < 4; i += 3){
	sum += i;
    }
    if(sum != 1) return 1;

    sum = 0;
#pragma acc parallel loop reduction(+:sum)
    for(i = 1; i < 5; i += 3){
	sum += i;
    }
    if(sum != 5) return 2;

    sum = 0;
#pragma acc parallel loop reduction(+:sum)
    for(i = 1; i < 6; i += 3){
	sum += i;
    }
    if(sum != 5) return 3;


    sum = 0;
#pragma acc parallel loop reduction(+:sum)
    for(i = 3; i > 0; i -= 3){
	sum += i;
    }
    if(sum != 3) return 4;

    sum = 0;
#pragma acc parallel loop reduction(+:sum)
    for(i = 4; i > 0; i -= 3){
	sum += i;
    }
    if(sum != 5) return 5;

    sum = 0;
#pragma acc parallel loop reduction(+:sum)
    for(i = 5; i > 0; i -= 3){
	sum += i;
    }
    if(sum != 7) return 6;

    printf("PASS\n");
    return 0;
}
