#include "data_extern.h"

int a[N];


void func()
{
    int i;
#pragma acc data pcopy(a)
#pragma acc parallel loop
    for(i=0;i<N;i++){
	a[i]++;
    }
}
