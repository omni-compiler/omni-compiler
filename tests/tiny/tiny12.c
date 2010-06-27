/*
 * OpenMP C API Test Suite
 * Example A.12 from OpenMP C/C++ API sepecification
 */

#include <stdio.h>

#define  N      613


int  x[2], y[N], idx[N];


void work(ip, jp)
    int  *ip, *jp;
{
    *ip = 1;
    *jp = 2;
}

void at001s(n, x, y, idx)
    int  n, *x, *y, *idx;
{
    int  i, xlocal, ylocal;

#pragma omp parallel for private(xlocal,ylocal) shared(x, y, idx, n)
    for( i = 0 ; i < n ; i++ ){
        work(&xlocal, &ylocal);
#pragma omp atomic
        x[idx[i]] += xlocal;
        y[i] += ylocal;
    }
}

main()
{
    int  i, errors, errors1, n;

    printf ("%x, %x, %x\n", x, y, idx);
    n = N;
    x[0] = 0;
    x[1] = 0;
    for( i = 0 ; i < n ; i++ ){
        y[i] = i;
        if ( i < n / 2 )
            idx[i] = 0;
        else
            idx[i] = 1;
    }
    at001s(n, x, y, idx);

    errors = 0;
    if ( x[0] != n/2 ){
        errors += 1;
        printf("atomic 001 - EXPECTED X[%d] = %d OBSERVED %d\n", 0, n/2, x[0]);
    }
    if ( x[1] != n - n/2 ){
        errors += 1;
        printf("atomic 001 - EXPECTED X[%d] = %d OBSERVED %d\n", 1, n-n/2, x[1]);
    }
    for( i = 0 ; i < n / 2 ; i++ ){
        if ( y[i] != 2+i )
            printf("atomic 001 - EXPECTED Y[%d] = %d OBSERVED %d\n", i, 2+i, y[i]);
    }

    if ( errors == 0 )
        printf("atomic 001 PASSED\n");
    else
        printf("atomic 001 FAILED\n");
}
