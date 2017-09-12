#include <stdio.h>

struct range{
    int lower;
    int length;
};

struct pos{
    int x;
    int y;
};

int test_struct_in_arrayref(void)
{
    struct range r = {1, 3};
    int i, a[10];

    for(i = 0; i < 10; i++){
	a[i] = 0;
    }

#pragma acc data copy(a[r.lower:r.length])
    {
#pragma acc parallel loop
	for(i = r.lower; i < r.lower + r.length; i++){
	    a[i] = i;
	}
    }

    for(i = 0; i < 10; i++){
	if(i >= r.lower && i < r.lower + r.length){
	    if(a[i] != i) return 1;
	}else{
	    if(a[i] != 0) return 2;
	}
    }

    return 0;
}

int not(int x) { return x? 0 : 1; }

int test_expr_in_if(void)
{
    int i, v[10];
    struct pos p = {0,3};
    struct pos *pp = &p;
    int a[4] = {0,1,2,3};
    int b[2][2] = {{0,1},{2,3}};

    for(i = 0; i < 10; i++){
	v[i] = 0;
    }

#pragma acc data copyin(v)
    {
#pragma acc parallel loop
	for(i = 0; i < 10; i++){
	    v[i] = 1;
	}
#pragma acc update host(v[0]) if(p.x)
#pragma acc update host(v[1]) if(pp->y)
#pragma acc update host(v[2]) if(a[1])
#pragma acc update host(v[3]) if(b[0][1])
#pragma acc update host(v[4]) if(not(1))
    }

    if(v[0] != 0) return 1;
    if(v[1] != 1) return 2;
    if(v[2] != 1) return 3;
    if(v[3] != 1) return 4;
    if(v[4] != 0) return 5;

    return 0;
}

int main(void)
{

    if(test_struct_in_arrayref()) return 1;
    if(test_expr_in_if()) return 2;

    printf("PASS\n");

    return 0;
}

