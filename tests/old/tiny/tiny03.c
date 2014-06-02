/*
 * OpenMP C API Test Suite
 * Example A.3 from OpenMP C/C++ API sepecification
 */

#include <stdio.h>
#include "omp.h"

#define NUM 1024


int  x[NUM+1];


void subdomain(x, iam, ipoints)
    int  *x, iam, ipoints;
{
    int  i, i1, i2;

    i1 = iam * ipoints + 1;
    i2 = i1 + ipoints - 1;
    for( i = i1 ; i <= i2 ; i++ )
        x[i] = iam;
}

void par001s(x, npoints)
    int  *x, npoints;
{
    int  iam, np, ipoints;

#pragma omp parallel shared(x, npoints) private(iam, np, ipoints)
    {
        iam = omp_get_thread_num();
        np = omp_get_num_threads();
        ipoints = npoints / np;
        subdomain(x, iam, ipoints);
    }
}

void main()
{

    const int  n = NUM;
    int  i, errors, ilast, nt;

    for( i = 1 ; i <= n ; i++ )
        x[i] = -1;

    par001s(x, n);
        /* Determine last element modified */
    ilast = 0;
    for( i = 1 ; i <= n ; i++ ){
        if ( x[i] < 0 )
            break;
        ilast = i;
    }
        /* Infer number of threads */
    nt = x[ilast] + 1;
    errors = 0;
        /* Should be fewer than NT points not modified */
    if ( n - ilast > nt ){
        errors += 1;
        printf("parallel - Threads do not divide points changed\n");
    }
    printf("parallel - Apparent number threads = %d\n", nt);
    if ( errors == 0 )
        printf("parallel 001 PASSED\n");
    else{
        printf("parallel -   Number points = %d\n", n);
        printf("parallel -   Points changed = %d\n", ilast);
        printf("parallel 001 FAILED\n");
    }
}
