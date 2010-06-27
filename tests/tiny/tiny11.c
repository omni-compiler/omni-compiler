/*
 * OpenMP C API Test Suite
 * Example A.11 from OpenMP C/C++ API sepecification
 */
void abort(void);
int omp_get_num_threads(void);
int omp_get_max_threads(void);
int omp_get_thread_num(void);
void omp_set_dynamic(int);
void omp_set_num_threads(int);


#include <stdio.h>


#define N       1024


int x[N];


void
do_by_x (x, iam, ipoints)
     int        x[];
     int        iam, ipoints;
{
  int   i;


  for (i = 0;  i < ipoints;  i ++) {
    x[iam*ipoints + i] = iam;
  }
}



void
test (x, npoints, thds)
     int        x[];
     int        npoints, thds;
{
  int   iam, ipoints;


  printf ("set thread = %d\n", thds);

  omp_set_dynamic (0);
  omp_set_num_threads (thds);
#pragma omp parallel shared (x, npoints) private (iam, ipoints)
  {
    if (omp_get_num_threads () != thds)
      abort ();

    iam = omp_get_thread_num ();
    ipoints = npoints/thds;
    do_by_x(x, iam, ipoints);
  }
}



int
check (x, npoints, thds)
     int        x[];
     int        npoints, thds;
{
  int   i, j, ipoints;

  int   errors = 0;


  ipoints = npoints / thds;
  for (i = 0;  i < thds;  i ++) {
    for (j = 0  ;  j < ipoints;  j ++) {
      if (x[i * ipoints + j] != i) {
        errors += 1;
        printf ("omp_set_num_threads - expected x[%d] = %d, observed %d\n",
                i * ipoints + j, i, x[i * ipoints + j]);
      }
    }
  }

  for (i = thds * ipoints;  i < npoints;  i ++) {
    if (x[i] != -1) {
      errors += 1;
      printf ("omp_set_num_threads - expected x[%d] = %d, observed %d\n",
              i, -1, x[i]);
    }
  }

  return errors;
}



main ()
{
  int   i, j, thds;

  int   errors = 0;


#ifdef __OMNI_SCASH__
  printf ("Omni on SCASH is not support omp_set_num_threads.\n");
  exit (1);
#endif

  thds = omp_get_max_threads ();
  for (i = 1;  i <= thds;  i ++) {
    for (j = 0;  j < N;  j ++) {
      x[j] = -1;
    }

    test (x, N, i);
    errors += check (x, N, i);
  }

  if (errors == 0) {
    printf ("omp_set_num_threads 001 PASSED\n");
  } else {
    printf ("omp_set_num_threads 001 FAILED\n");
  }
}
