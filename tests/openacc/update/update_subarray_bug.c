#include <stdio.h>
#define N 1000

int main(void)
{
    int a[N];
    int i;

    for(i = 0;i < N; i++){
	a[i] = i;
    }

#pragma acc enter data copyin(a)

#pragma acc parallel loop
    for(i = 0;i < N; i++){
	a[i] += 2;
    }

    int n_half = N/2;

#pragma acc data present(a[n_half:n_half])
    {
#pragma acc update host(a[n_half:n_half])
    }

    for(i = 0;i < n_half; i++){
	if(a[i] != i) return 1;
    }
    for(i = n_half;i < N; i++){
	if(a[i] != i + 2) return 2;
    }

    printf("PASS\n");
    return 0;
}
