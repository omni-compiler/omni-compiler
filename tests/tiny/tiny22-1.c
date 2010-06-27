/*
 * OpenMP C API Test Suite
 * Example A.22 from OpenMP C/C++ API sepecification
 */
#include <stdio.h>
#include <omp.h>


#define N       1024


int     a[N];
int     m[N], n[N];
int     v[N], w[N], x[N], y[N], z[N];
int     s = 1;


#pragma omp threadprivate(v)


void
test (a)
     int        a;
{
  const int     c = 2;
  int           i = 3;


#pragma omp parallel default(none) private(a) shared(m,n,w,x,y,z)
  {
    int j = omp_get_thread_num ();    /* OK. - j is declared within parallel region */

    a    = j + 1;                     /* OK. - a is listed in private clause */
    v[j] = a;                         /* OK. - v is threadprivate */
    z[j] = v[j];
    w[j] = j + 1;                     /* OK. - w is listed in shared clause */
    x[j] = c;                         /* OK. - c has const-qualified type */
#if 0
    m[j] = s;                         /* NG  - cannot reference i or s here */
#endif

#pragma omp for firstprivate(s)
    for (i=0;  i<10;  i++) {
      y[i] = s;                       /* OK. = i is the loop control variable */
                                      /*       s is listed in firstprivate clause */
    }
#if 0
    n[j] = s;                         /* NG  - cannot reference i or s here */
#endif

  }
}



main ()
{
  int   thds, i;
  
  int   errors = 0;


  test (-1);

  thds = omp_get_max_threads ();
  for (i=0;  i<thds;  i++) {
    if (z[i] != i+1) {
      errors += 1;
      printf ("default(none) - expected z[%d] = %d, observed %d\n",
              i, i+1, z[i]);
    }
  }

  for (i=0;  i<thds;  i++) {
    if (w[i] != i+1) {
      errors += 1;
      printf ("default(none) - expected w[%d] = %d, observed %d\n",
              i, i+1, w[i]);
    }
  }

  for (i=0;  i<thds;  i++) {
    if (x[i] != 2) {
      errors += 1;
      printf ("default(none) - expected x[%d] = %d, observed %d\n",
              i, 2, x[i]);
    }
  }

  for (i=0;  i<10;  i++) {
    if (y[i] != s) {
      errors += 1;
      printf ("default(none) - expected y[%d] = %d, observed %d\n",
              i, s, y[i]);
    }
  }
  for (i=10;  i<N;  i++) {
    if (y[i] != 0) {
      errors += 1;
      printf ("default(none) - expected y[%d] = %d, observed %d\n",
              i, 0, y[i]);
    }
  }



  if (errors == 0) {
    printf ("default(none) PASSED\n");
  } else {
    printf ("default(none) FAILED\n");
  }
}
