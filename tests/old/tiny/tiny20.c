/*
 * OpenMP C API Test Suite
 * Example A.20 from OpenMP C/C++ API sepecification
 */
void exit(int);
#include <stdio.h>
#include <omp.h>


#define N       1024


int             x[N], y[N];
omp_lock_t      lckx, lcky;
int             nest;
int             cmpl;


void
work1 (n)
     int        n;
{
  while(nest != 0  &&  n == 0  &&  cmpl == 0) {
#pragma omp flush
  }

  omp_set_lock (&lckx);
#pragma omp flush
  x[n] += n;
  y[n] = x[n];
#pragma omp flush
  omp_unset_lock (&lckx);
}


void
work2 (n)
     int        n;
{
  if (nest != 0  &&  n == 1) {
    cmpl = 1;
#pragma omp flush
  }

  omp_set_lock (&lcky);
#pragma omp flush
  y[n] -= n;
#pragma omp flush
  omp_unset_lock (&lcky);
}


void
sub3 (n)
     int        n;
{
  work1 (n);
#pragma omp barrier
  work2 (n);
}



void
sub2 (k)
     int        k;
{
#pragma omp parallel shared (k)
  sub3(k);
}



void
sub1 (n)
     int        n;
{
  int   i;


#pragma omp parallel private(i) shared(n)
  {
#pragma omp for
    for (i=0;  i<n;  i++) {
      sub2(i);
    }
  }
}


void
init (nf)
     int        nf;
{
  int   i;


  nest = nf;
  cmpl = 0;

  for (i = 0;  i<N;  i++) {
    x[i] = 0;
    y[i] = -1;
  }
}


int main ()
{
  int   i, thds;

  int   errors = 0;


  thds = omp_get_max_threads();
  if (thds == 1) {
    printf ("This test program can not execute 1 thread.\n");
    printf ("Plese set OMP_NUM_THREADS > 1, and try again.\n");
    exit(0);
  }



  omp_init_lock (&lckx);
  omp_init_lock (&lcky);


  /* OMPC_NESTED=true, and nested parallel case */
  init (1);
  omp_set_nested (1);
  if (omp_get_nested() == 0) {
    printf ("skip nested case test\n");
  } else {
    sub1 (2);
    thds = omp_get_max_threads ();
    for (i=0;  i<N;  i++) {
      if (i < 2) {
        if (x[i] == 0) {
          errors += 1;
          printf ("E01:OMPC_NESTED=true nested case - expected x[%d] != %d, observed %d\n",
                  0, i*thds, x[i]);
        }
        if (y[i] != 0) {
          errors += 1;
          printf ("E02:OMPC_NESTED=true nested case - expected y[%d] = %d, observed %d\n",
                  i, 0, y[i]);
        }
      } else {
        if (x[i] != 0) {
          errors += 1;
          printf ("E03:OMPC_NESTED=true nested case - expected x[%d] = %d, observed %d\n",
                  i, 0, x[i]);
        }
        if (y[i] != -1) {
          errors += 1;
          printf ("E04:OMPC_NESTED=true nested case - expected y[%d] = %d, observed %d\n",
                  i, -1, y[i]);
        }
      }
    }
  }


  /* OMPC_NESTED=false, and nested parallel case */
  init (0);
  omp_set_nested (0);
  sub1 (2);

  thds = omp_get_max_threads ();
  for (i=0;  i<N;  i++) {
    if (i < 2) {
      if (x[i] != i) {
        errors += 1;
        printf ("E05:OMPC_NESTED=false nested case - expected x[%d] = %d, observed %d\n",
                i, i, x[i]);
      }
      if (y[i] != 0) {
        errors += 1;
        printf ("E06:OMPC_NESTED=false nested case - expected y[%d] = %d, observed %d\n",
                i, 0, y[i]);
      }
    } else {
      if (x[i] != 0) {
        errors += 1;
        printf ("E07:OMPC_NESTED=false nested case - expected x[%d] = %d, observed %d\n",
                i, 0, x[i]);
      }
      if (y[i] != -1) {
        errors += 1;
        printf ("E08:OMPC_NESTED=false nested case - expected y[%d] = %d, observed %d\n",
                i, -1, y[i]);
      }
    }
  }


  /* test : not nested case */
  init (0);
  sub2 (2);

  thds = omp_get_max_threads ();
  for (i=0;  i<N;  i++) {
    if (i == 2) {
      if (x[i] != 2*thds) {
        errors += 1;
        printf ("E09:not nested case - expected x[%d] = %d, observed %d\n",
                i, 2*thds, x[i]);
      }
      if (y[i] != 0) {
        errors += 1;
        printf ("E10:not nested case - expected y[%d] = %d, observed %d\n",
                i, 0, y[i]);
      }
    } else {
      if (x[i] != 0) {
        errors += 1;
        printf ("E11:not nested case - expected x[%d] = %d, observed %d\n",
                i, 0, x[i]);
      }
      if (y[i] != -1) {
        errors += 1;
        printf ("E12:not nested case - expected y[%d] = %d, observed %d\n",
                i, -1, y[i]);
      }
    }
  }


  omp_destroy_lock (&lckx);
  omp_destroy_lock (&lcky);


  if (errors == 0) {
    printf ("nested parallel region PASSED\n");
  } else {
    printf ("nested parallel region FAILED\n");
  }
}
