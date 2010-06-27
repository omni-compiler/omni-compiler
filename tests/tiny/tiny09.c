/*
 * OpenMP C API Test Suite
 * Example A.9 from OpenMP C/C++ API sepecification
 */

#include <stdio.h>

int  x, y;


void output(x)
    int  *x;
{
    *x += 10;
}

void input(y)
    int  *y;
{
    *y += 100;
}

void work(x)
    int  *x;
{
#pragma omp atomic
    *x += 1;
}

void sngl001s(x, y)
    int  *x, *y;
{
#pragma omp parallel default(shared)
    {
        work(x);
#pragma omp barrier
#pragma omp single
        {
            output(x);
            input(y);
        }
        work(y);
    }
}

main()
{
    int  i, errors;

    x = 0;
    y = -100;
    sngl001s(&x, &y);

    errors = 0;
    if ( x <= 0 ){
        errors += 1;
        printf("single 001 - Expect positive value, observe X = %d\n", x);
    }
    if ( y <= 0 ){
        errors += 1;
        printf("single 001 - Expect positive value, observe Y = %d\n", y);
    }
    else
        printf("single 001 - Apparent number of threads, Y = %d\n", y);

    if ( x - y != 10 ){
        errors += 1;
        printf("single 001 - Expect difference of 10, observe X = %d, Y = %d\n",
               x, y);
    }

    if ( errors == 0 )
        printf("single 001 PASSED\n");
    else
        printf("single 001 FAILED\n");
}
