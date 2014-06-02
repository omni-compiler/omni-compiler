/* incorrect case
 * OpenMP C API Test Suite
 * Example A.22 from OpenMP C/C++ API sepecification
 */
#include <stdio.h>
#include <omp.h>


#define N       1024


int     n[N];
int     y[N];
int     s = 1;


void
test ()
{
  int           i = 3;


#pragma omp parallel default(none) shared(n,y)
  {
    int j = omp_get_thread_num ();

#pragma omp for firstprivate(s)
    for (i=0;  i<10;  i++) {
      y[i] = s;
    }
    n[j] = s;                         /* NG  - cannot reference i or s here */
  }
}



main ()
{
  int   thds, i;
  
  int   errors = 0;


  test ();

  thds = omp_get_max_threads ();
  for (i=0;  i<thds;  i++) {
    if (n[i] != s) {
      errors += 1;
      printf ("default(none) - expected n[%d] = %d, observed %d\n",
              i, s, n[i]);
    }
  }

  if (errors == 0) {
    printf ("default(none) PASSED\n");
  } else {
    printf ("default(none) FAILED\n");
  }
}
