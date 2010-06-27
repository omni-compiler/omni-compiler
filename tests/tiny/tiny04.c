/*
 * OpenMP C API Test Suite
 * Example A.4 from OpenMP C/C++ API sepecification
 */

#include <stdio.h>

#define M       117
#define N       511

int  a[N], b[N], y[M], z[M];


int sqrt_func(k)
    int  k;
{
    int  i, j;

    j = 1;
    for( i = 0 ; i < k ; i++ ){
        if ( j*j != k )
            j = (j + k/j) / 2;
        else
            break;
    }
    return (j);
}

void d001s(m, n, a, b, y, z)
    int  m, n, *a, *b, *y, *z;
{
    int  i;

#pragma omp parallel
    {
#pragma omp for nowait
        for( i = 1 ; i < n ; i++ )
            b[i] = (a[i] + a[i-1]) / 2;
#pragma omp for nowait
        for( i = 1 ; i <= m ; i++ )
            y[i] = sqrt_func(z[i]);
    }
}

main()
{
    int  m, n, i, errors;

    m = M;
    n = N;
    for( i = 0 ; i < n ; i++ ){
        a[i] = 2*i + 1;
        b[i] = 0;
    }
    for( i = 0 ; i < m ; i++ ){
        z[i] = i*i;
        y[i] = 0;
    }

    d001s(m, n, a, b, y, z);

    errors = 0;
    for( i = 1 ; i < n ; i++ ){
        if ( b[i] != 2*i ){
            errors += 1;
            if ( errors == 1 )
                printf("for - VALUES IN B ARE NOT AS EXPECTED\n");

            printf("EXPECTED B[%d] = %d OBSERVED %d\n", i, 2*i, b[i]);
        }
    }
    for( i = 1 ; i < m ; i++ ){
        if ( y[i] != i ){
            errors += 1;
            if ( errors == 1 )
                printf("for - VALUES IN Y ARE NOT AS EXPECTED\n");

            printf("EXPECTED Y[%d] = %d OBSERVED %d\n", i, i, y[i]);
        }
    }
    if ( errors == 0 )
        printf("for 001 PASSED\n");
    else
        printf("for 001 FAILED\n");
}
