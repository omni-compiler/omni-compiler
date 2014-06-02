/*
 * OpenMP C API Test Suite
 * Example A.15 from OpenMP C/C++ API sepecification
 */
int omp_get_num_threads(void);
int omp_get_max_threads(void);
int omp_get_thread_num(void);

#include <stdio.h>


#define N       1024

int     x[N], y[N];


void
work (i)
     int        i;
{
  x[i] = i;
}


void
work2 (i)
     int        i;
{
  y[i] = i;
}


void
test1 ()
{
  int   i, np;


  np = omp_get_num_threads ();
#pragma omp parallel for schedule(static)
  for (i=0;  i<np;  i++) {
    work (i);
  }
}


void
test2 ()
{
  int   i;


#pragma omp parallel private(i)
  {
    i = omp_get_thread_num ();
    work2 (i);
  }
}



main ()
{
  int   i, thds;

  int   errors = 0;


  for (i=0;  i<N;  i++) {
    x[i] = -1;
    y[i] = -1;
  }

  test1 ();
  test2 ();

  thds = 1;
  for (i=0; i<thds; i++) {
    if (x[i] != i) {
      printf ("omp_get_thread_num 001 - expected x[%d] = %d, observed %d\n",
              i, i, x[i]);
      break;
    }
  }
  for (; i<N; i++) {
    if (x[i] != -1) {
      printf ("omp_get_thread_num 001 - expected x[%d] = %d, observed %d\n",
              i, -1, x[i]);
      break;
    }
  }
  for (thds=0; thds<N; thds++) {
    if (y[thds] != thds) {
      break;
    }
  }
  for (i=thds; i<N; i++) {
    if (y[i] != -1) {
      printf ("omp_get_thread_num 001 - expected y[%d] = %d, observed %d\n",
              i, -1, y[i]);
      break;
    }
  }

  printf ("this program run on %d theads\n", thds);

  if (errors == 0) {
    printf ("omp_get_thread_num 001 : PASSED\n");
  } else {
    printf ("omp_get_thread_num 001 : FAILED\n");
  }
}
