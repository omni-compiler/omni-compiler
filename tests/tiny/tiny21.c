/*
 * OpenMP C API Test Suite
 * Example A.18 from OpenMP C/C++ API sepecification
 */
#include <stdio.h>


main()
{
    int  i, j, k;


    i = 1;
    j = 2;
    k = 0;
#pragma omp parallel private(i) firstprivate(j)
    {
        i = 3;
        j += 2;
#pragma omp atomic
        k += 1;
    }
    printf("par002 - values of I: %d, J: %d, and K: %d\n", i, j, k);


    if ( k < 1 )
        printf("parallel 002 - NOTE serial or team size of one\n");
    if ( i == 1 && j == 2 )
        printf("parallel 002 - NOTE original variable retains original value\n");
    if ( i == 3 ){
        printf("parallel 002 - NOTE original variable gets value of master\n copy of private variable\n");
        if ( j == 4 )
            printf("parallel 002 - NOTE value of J gives no evidence\n of parallel execution\n");
    }
    printf("parallel 002 PASSED\n");
}
