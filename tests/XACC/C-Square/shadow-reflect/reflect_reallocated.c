#include <stdio.h>
#define N 1024

#pragma xmp nodes p[*]
#pragma xmp template t[N]
#pragma xmp distribute t[block] onto p

int main(void)
{
    int a[N], b[N];
#pragma xmp align a[i] with t[i]
#pragma xmp align b[i] with t[i]
#pragma xmp shadow a[1:1]
#pragma xmp shadow b[1:1]
    void *acc_a_0, *acc_a_1;

#pragma acc data copy(a)
    {
#pragma xmp loop [i] on t[i]
#pragma acc parallel loop
	for(int i = 0; i < N; i++){
	    a[i] = i;
	}
#pragma xmp reflect(a) acc

#pragma acc host_data use_device(a)
	acc_a_0 = a;
    }

#pragma acc enter data create(b)
#pragma acc enter data copyin(a)

#pragma acc data present(a,b)
    {
#pragma acc host_data use_device(a)
	acc_a_1 = a;

	if(acc_a_0 == acc_a_1){
#pragma xmp task on p[0]
	    printf("Cannot test reflect because device addresses of array 'a' are same");

	    return 2;
	}

#pragma xmp loop [i] on t[i]
#pragma acc parallel loop
	for(int i = 0; i < N; i++){
	    a[i] += i;
	}

//expect that reflect communication is rescheduled because device memory is reallocated.
#pragma xmp reflect(a) acc

#pragma xmp loop [i] on t[i]
#pragma acc parallel loop
	for(int i = 1; i < N-1; i++){
	    b[i] = a[i-1] + a[i+1];
	}
    }
#pragma acc exit data copyout(b) delete(a)

    int err = 0;
#pragma xmp loop [i] on t[i] reduction(+:err)
    for(int i = 1; i < N-1; i++){
	if(b[i] != i * 4) err++;
    }

    if(err > 0){
#pragma xmp task on p[0]
	printf("Failed\n");
	return 1;
    }

#pragma xmp task on p[0]
    printf("PASS\n");

    return 0;
}
