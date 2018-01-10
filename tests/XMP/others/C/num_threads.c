#include <stdio.h>
#include <omp.h>

int f(int aBc)
{
    int numt;
#pragma omp parallel num_threads(aBc)
    {
        numt = omp_get_max_threads();

#pragma omp barrier

    }

    return 5 - printf((numt == 4) ? "PASS\n" : "ERROR\n");
}

int main () {
    return f(4);
}

