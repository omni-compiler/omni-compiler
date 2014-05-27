/*
 * OpenMP C API Test Suite
 * Example A.18 from OpenMP C/C++ API sepecification
 */
#include <stdio.h>

#define  N      117

int  iw[N][N];

void work(i, j)
    int  i, j;
{
    iw[i][j] = i + j * N;
}

void some_work(i, n)
    int  i, n;
{
    int  j;
#pragma omp parallel default(shared)
    {
#pragma omp for
        for( j = 0 ; j < n ; j++ )
            work(i, j);
    }
}

void d003s(n)
    int  n;
{
    int  i, j;
#pragma omp parallel default(shared)
    {
#pragma omp for
        for( i = 0 ; i < n ; i++ ){
#pragma omp parallel shared(i, n)
#pragma omp for
            for( j = 0 ; j < n ; j++ )
                work(i, j);
        }
    }
}

void d003t(n)
    int  n;
{
    int  i;
#pragma omp parallel default(shared)
    {
#pragma omp for
        for( i = 0 ; i < n ; i++ )
            some_work(i, n);
    }
}

void init()
{
    int  i, j;
    for( i = 0 ; i < N ; i++ )
        for( j = 0 ; j < N ; j++ )
            iw[i][j] = 0;
}

int icheck(ch)
    char  * ch;
{
    int  i, j, iexpect, res = 0;

    for( i = 0 ; i < N ; i++ )
        for( j = 0 ; j < N ; j++ ){
            iexpect = i + j * N;
            if ( iw[i][j] != iexpect ){
                res += 1;
                printf("for 003 - EXPECTED IW[%d][%d] = %d OBSERVED %d %s\n",
                       i, j, iexpect, iw[i][j], ch);
            }
        }

    return res;
}

main()
{
    int  i, n, errors;

    n = N;
    errors = 0;
    init();
    d003s(n);
    errors += icheck("after d003s");

    init();
    d003t(n);
    errors += icheck("after d003t");

    if ( errors == 0 )
        printf("for 003 PASSED\n");
    else
        printf("for 003 FAILED\n");
}
