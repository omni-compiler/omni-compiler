#include <stdio.h>

#define ARRAYSIZE 10

int errors = 0;

#if _OPENMP
extern int omp_get_thread_num (void);
#endif

void f(int n, int a[*]);

void f(int n, int a[n])
{
    #pragma omp parallel private(a)
    {
#if _OPENMP
    int id = omp_get_thread_num ();
#else
    int id = 0;
#endif

    for(int i = 0; i < n; i++)
        a[i] = id;

    #pragma omp barrier

    for(int i = 0; i < n; i++)
        if (a[i] != id) {
            #pragma omp critical
            errors += 1;
        }
    }
}

int main ()
{
    int a[ARRAYSIZE];

    f(ARRAYSIZE, a);

    if (errors == 0) {
        printf ("vla 002 : SUCCESS\n");
        return 0;
    } else {
        printf ("vla 002 : FAILED\n");
        return 1;
    }
}

