/*
 * OpenMP C API Test Suite
 * Example A.13 from OpenMP C/C++ API sepecification
 */
int omp_get_num_threads(void);
int omp_get_max_threads(void);
int omp_get_thread_num(void);

#include <stdio.h>

#define N       1024


int     x[N], sync[N], work[N];


tests (x)
     int        x[];
{
  int   i, iam, neighbor;


  for (i=0;  i<N;  i++) {
    sync[i] = 0;
    work[i] = -1;
  }

#pragma omp parallel private(iam,neighbor) shared(work,sync)
  {
    iam = omp_get_thread_num ();
    sync[iam] = 0;
#pragma omp barrier

    /* Do computation into my portion of work array */
    work[iam] = iam;
#pragma omp flush (work)

    sync[iam] = 1;
#pragma omp flush (sync)
    neighbor = (iam >0 ? iam  : omp_get_num_threads()) - 1;
    while (sync[neighbor] == 0) {
#pragma omp flush (sync)
    }
    
    /* Read neighbor's values of work array */
#pragma omp flush (work)
    x[iam] = work[neighbor];
  }
}


main ()
{
  int   i, thds, v;

  int   errors = 0;

  for (i=0;  i<N;  i++) {
    x[i] = -1;
  }

  tests (x);

  thds = omp_get_max_threads ();
  for (i=0;  i<thds;  i++) {
    v = (i > 0 ? i : thds) - 1;
    if (x[i] != v) {
      errors += 1;
      printf ("flush 001 - expected x[%d] = %d, observed %d\n",
              i, v, x[i]);
    }
  }

  if (errors == 0) {
    printf ("flush 001 PASSED\n");
  } else {
    printf ("flush 001 FAILED\n");
  }
}

