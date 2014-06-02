/*
 * OpenMP C API Test Suite
 * Example A.7 from OpenMP C/C++ API sepecification
 */

#include <stdio.h>

#define  N      1024

void work(ip, jp)
    int  *ip, *jp;
{
    *ip = 1;
    *jp = -1;
}

void pd002s(n, a, b)
    int  n, *a, *b;
{
    int  i, alocal, blocal, x, y;

    x = 0;
    y = 0;
#pragma omp parallel for private(i, alocal, blocal) shared(n) reduction(+: x, y)
    for( i = 0 ; i < n ; i++ ){
        work(&alocal, &blocal);
        x += alocal;
        y += blocal;
    }

    *a += x;
    *b += y;
}

main()
{
    int  i, n, errors;
    int  a, b;

    n = N;
    a = 0;
    b = n;
    pd002s(n, &a, &b);

    errors = 0;
    if ( a != n ){
        errors += 1;
        printf("parallel for 002 - EXPECTED A = %d OBSERVED %d\n", n, a);
    }
    if ( b != 0 ){
        errors += 1;
        printf("parallel for 002 - EXPECTED B = %d OBSERVED %d\n", 0, b);
    }

    if ( errors == 0 )
        printf("parallel for 002 PASSED\n");
    else
        printf("parallel for 002 FAILED\n");
}
