#include "data_extern.h"

int main(void)
{
    int i;
#pragma acc data copy(a)
    {
#pragma acc parallel loop
	for(i=0;i<N;i++){
	    a[i] = i;
	}
	func();
    }
    for(i=0;i<N;i++){
	if(a[i] != i+1) return 1;
    }

    printf("PASS\n");
    return 0;
}
