#include <stdio.h>

#define ARRAYSIZE 10

int errors = 0;
int org[10] = {1,2,3,4,5,6,7,8,9,10};

void f(int n, int a[*]);

void f(int n, int a[n])
{
    for(int i = 0; i < n; i++)
    {
        if(a[i] != org[i]) {
            #pragma omp critical
            errors++;
        }
    }
}

int main ()
{
    #pragma omp parallel
    f(10,org);

    if (errors == 0) {
        printf ("vla 001 : SUCCESS\n");
        return 0;
    } else {
        printf ("vla 001 : FAILED\n");
        return 1;
    }
}

