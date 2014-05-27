/*
 * OpenMP C API Test Suite
 * Example A.16 from OpenMP C/C++ API sepecification
 */
#include <stdio.h>
#include <omp.h>


#define N       1024


omp_lock_t      lck;

int     x[N];
int     y[N];


void
work (id)
     int        id;
{
  static int    xnum = 0;

#pragma omp flush
  x[xnum++] = id;
#pragma omp flush
}


void
work2 (id)
     int        id;
{
  static int    ynum = 0;

#pragma omp flush
  y[ynum++] = id;
#pragma omp flush
}


void
skip (id)
     int        id;
{
  printf ("thread %d is skip\n", id);
}


void
test ()
{
  int           id;

  int           i = 0;


  omp_init_lock (&lck);
#pragma omp parallel shared (lck,i) private (id)
  {
    int cnt = 0;
    id = omp_get_thread_num ();

    omp_set_lock (&lck);
    printf ("My thread id is %d.\n", id);
    work2 (id);
    omp_unset_lock (&lck);

    while (! omp_test_lock (&lck)) {
      if (cnt++ == 0) {
        skip (id);
      }
      /* we do not yet have the lock,
       * so we must do something else
       */
    }
    work (id);
    omp_unset_lock (&lck);
  }

  omp_destroy_lock (&lck);
}


main ()
{
  int   i, j, thds, chk;

  int   errors = 0;


  for (i=0;  i<N;  i++) {
    x[i] = -1;
    y[i] = -1;
  }

  test ();


  thds = omp_get_max_threads ();
  for (i=0;  i<thds;  i++) {
    chk = 0;
    for (j=0;  j<thds;  j++) {
      if (x[j] == i) {
        chk = 1;
      }
    }
    if (chk == 0) {
      errors += 1;
      printf ("omp_lock 001 - expected x[?] = %d, not exist\n",
              i);
    }
  }
  for (i=thds;  i<N;  i++) {
    if (x[i] != -1) {
      errors += 1;
      printf ("omp_lock 001 - expected x[%d] = %d, observed %d\n",
              i, -1, x[i]);
    }
  }

  for (i=0;  i<thds;  i++) {
    chk = 0;
    for (j=0;  j<thds;  j++) {
      if (y[j] == i) {
        chk = 1;
      }
    }
    if (chk == 0) {
      errors += 1;
      printf ("omp_lock 001 - expected y[?] = %d, not exist\n",
              i);
    }
  }
  for (i=thds;  i<N;  i++) {
    if (y[i] != -1) {
      errors += 1;
      printf ("omp_lock 001 - expected y[%d] = %d, observed %d\n",
              i, -1, y[i]);
    }
  }

  if (errors == 0) {
    printf ("omp_lock 001 PASSED\n");
  } else {
    printf ("omp_lock 001 FAILED\n");
  }
}
