/*
 * OpenMP C API Test Suite
 * Example A.8 from OpenMP C/C++ API sepecification
 */

#include <stdio.h>

int  axis[3];

void msg(who, count)
    char  * who;
    int  count;
{
    printf("%s: MESSAGE %d\n", who, count);
}

void xaxis()
{
    axis[0] = 1;
    msg("xaxis", 1);
    msg("xaxis", 2);
    msg("xaxis", 3);
}

void yaxis()
{
    axis[1] = 1;
    msg("yaxis", 1);
    msg("yaxis", 2);
    msg("yaxis", 3);
}

void zaxis()
{
    axis[2] = 1;
    msg("zaxis", 1);
    msg("zaxis", 2);
    msg("zaxis", 3);
}

void ps001s()
{
#pragma omp parallel sections
    {
#pragma omp section
        xaxis();
#pragma omp section
        yaxis();
#pragma omp section
        zaxis();
    }
}

main()
{
    int  i, errors;
        /* Element 1 is the index of the next item to be served */
    ps001s();
        /* Check elements of queue */
    errors = 0;
    for( i = 0 ; i < 3 ; i++ ){
        if ( axis[i] != 1 ){
            errors += 1;
            printf("parallel section 001 - expected AXIS[%d] = 1, observed %d\n",
                   i, axis[i]);
        }
    }

    if ( errors == 0 )
        printf("parallel section 001 PASSED\n");
    else
        printf("parallel section 001 FAILED\n");
}
