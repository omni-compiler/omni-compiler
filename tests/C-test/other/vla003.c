#include <stdio.h>

#define ARRAYSIZE 10

int errors = 0;

#if _OPENMP
extern int omp_get_thread_num (void);
extern int omp_get_num_threads (void);
#endif

void f(int n, int a[n])
{
    #pragma omp parallel shared(a)
    {
#if _OPENMP
        int id = omp_get_thread_num ();
        int max = omp_get_num_threads ();
#else
        int id = 0;
        int max = n;
#endif

        #pragma omp critical
        {
            if(id < n)
                a[id] = id;
        }

        #pragma omp barrier

        for(int i = 0; i < n && i < max; i++) {
            if (a[i] != i) {
                #pragma omp critical
                errors++;
            }
        }
    }
}

int main ()
{
    int a[10];

    f(10, a);

    if (errors == 0) {
        printf ("vla 003 : SUCCESS\n");
        return 0;
    } else {
        printf ("vla 003 : FAILED\n");
        return 1;
    }
}

