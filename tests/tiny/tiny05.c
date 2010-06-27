/*
 * OpenMP C API Test Suite
 * Example A.5 from OpenMP C/C++ API sepecification
 */
int omp_get_max_threads();

#include <stdio.h>


#define N       1024

int     x[N];
int     y[N];


int *
dequeue (a)
     int        a[];
{
  int   *ret;

  ret   = &a[a[0]];
  a[a[0]] = 1;
  a[0] += 1;

  return ret; 
}


void
work (a)
     int        *a;
{
  *a += 1;
}


void
critical (x, y)
     int        x[], y[];
{
  int   *x_next, *y_next;


#pragma omp parallel  shared(x,y) private(x_next, y_next)
  {
#pragma omp critical(xaxis)
    x_next = dequeue (x);
    work (x_next);
#pragma omp critical(yaxis)
    y_next = dequeue (y);
    work (y_next);
  }
}


main ()
{
  int   i;

  int   errors = 0;
  int   thds;


  for (i = 1;  i < N;  i ++) {
    x[i] = -1;
    y[i] = -1;
  }
  x[0] = 1;
  y[0] = 10;


  critical (x, y);


  thds = omp_get_max_threads ();
  for (i = 1;  i < N;  i ++) {
    if ((x[0] - thds <= i) && (i < x[0])) {
      if (x[i] != 2) {
        errors += 1;
        printf ("critical - expected x[%d] = %d, observed %d\n",
                i, 2, x[i]);
      }
    } else {
      if (x[i] != -1) {
        errors += 1;
        printf ("critical - expected x[%d] = %d, observed %d\n",
                i, -1, x[i]);
      }
    }

    if ((y[0] - thds <= i) && (i < y[0])) {
      if (y[i] != 2) {
        errors += 1;
        printf ("critical - expected y[%d] = %d, observed %d\n",
                i, 2, y[i]);
      }
    } else {
      if (y[i] != -1) {
        errors += 1;
        printf ("critical - expected y[%d] = %d, observed %d\n",
                i, -1, y[i]);
      }
    }
  }


  if (errors == 0) {
    printf ("critical 001 PASSED\n");
  } else {
    printf ("critical 001 FAILED\n");
  }
}
