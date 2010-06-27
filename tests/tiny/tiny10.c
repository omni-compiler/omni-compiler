/*
 * OpenMP C API Test Suite
 * Example A.10 from OpenMP C/C++ API sepecification
 */
#include <stdio.h>


#define N       1024


int     x[N];
int     pos;


void
work (int k)
{
#pragma omp ordered
  {
    printf (" %d", k);
    x[pos++] = k;
  }
}


test (lb, ub, st)
     int        lb, ub, st;
{
  int   i;


  printf ("output in parallel regeon : ");
#pragma omp parallel
  {
#pragma omp for ordered schedule(dynamic)
    for (i=lb;  i<ub;  i+=st) {
      work (i);
    }
  }
  printf ("\n");

  printf ("output in sequential regeon : ");
  {
#pragma omp for ordered schedule(dynamic)
    for (i=lb;  i<ub;  i+=st) {
      work (i);
    }
  }
  printf ("\n");
}



main ()
{
  int   i;

  int   errors = 0;


  for (i = 0;  i < N;  i++) {
    x[i] = -1;
  }
  pos = 0;


  test (1, 30, 2);

  pos = 0;
  for (i = 1;  i < 30;  i +=2) {
    if (x[pos] != i) {
      errors += 1;
      printf ("ordered - expected x[%d] = %d, observed %d\n",
              pos, i, x[pos]);
    }
    pos ++;
  }

  for (i = 1;  i < 30;  i +=2) {
    if (x[pos] != i) {
      errors += 1;
    if (x[pos] != i) {
      errors += 1;
      printf ("ordered 001 - expected x[%d] = %d, observed %d\n",
              pos, i, x[pos]);
    }
    }
    pos ++;
  }

  for (;  pos < N; pos ++) {
    if (x[pos] != -1) {
      errors += 1;
    if (x[pos] != i) {
      errors += 1;
      printf ("ordered 001 - expected x[%d] = %d, observed %d\n",
              pos, -1, x[pos]);
    }
    }
    pos ++;
  }


  if (errors == 0) {
    printf ("ordered 001 PASSED\n");
  } else {
    printf ("ordered 001 FAILED\n");
  }
}



