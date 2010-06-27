/*
 * OpenMP C API Test Suite
 * Example A.6 from OpenMP C/C++ API sepecification
 */

#include <stdio.h>

#define  N      31

int  a[N], b[N], c[N];


void reverse(int);

void d002s(n, a, b, c)
    int  n, *a, *b, *c;
{
    int i;

#pragma omp parallel
    {
#pragma omp for lastprivate(i)
        for( i = 0 ; i < n ; i++ )
            a[i] = b[i] + c[i];
    }

    reverse(i);
}

static int gj = 0;

void reverse(i)
    int  i;
{
    gj = i;
}

main()
{
    int  i, n, errors;

    n = N;
    for( i = 0 ; i < n ; i++ ){
        a[i] = 0;
        b[i] = i;
        c[i] = 2*i;
    }

    d002s(n, a, b, c);

    errors = 0;
    if ( gj != n ){
        errors += 1;
        printf("for 002 - EXPECTED J = %d OBSERVED %d\n", n, gj);
    }

    for( i = 0 ; i < n ; i++ ){
        if ( a[i] != i*3 ){
            errors += 1;
            if ( errors == 1 )
                printf("for002 - VALUES IN A ARE NOT AS EXPECTED\n");
            printf("EXPECTED A(%d) = %d OBSERVED %d\n", i, 3*i, a[i]);
        }
    }
    if ( errors == 0 )
        printf("for 002 PASSED\n");
    else
        printf("for 002 FAILED\n");
}
