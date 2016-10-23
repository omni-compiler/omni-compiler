#include <stdio.h>
#define N 200

void hoge(int *a, int n)
{
    int i;
#pragma acc parallel loop deviceptr(a[0:n])
    for(i = 0; i < n; i++){
	a[i] *= 2;
    }
}

int main(void)
{
    int a[N];
    int i;
    for(i = 0; i < N; i++){
	a[i] = i + 1;
    }
#pragma acc enter data copyin(a[0:N])

#pragma acc host_data use_device(a)
    hoge(a, N);

#pragma acc exit data copyout(a)

    for(i = 0; i < N; i++){
	if(a[i] != (i + 1)*2) return 1;
    }

    printf("PASS\n");
    return 0;
}
