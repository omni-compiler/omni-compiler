/* incorrect case
 * OpenMP C API Test Suite
 * Example A.19 from OpenMP C/C++ API sepecification
 */
#include <stdio.h>

#define N       1024

int     x[N];

work (i)
     int        i;
{
  x[i] = i;
}


main ()
{
  int   j;

  int   n = N;
  int   errors = 0;


  printf ("!!! this program is wrong.\n"
          "!!! always FAILED or compile error\n");

  for (j=0;  j<N;  j++) {
    x[j] = -1;
  }

#pragma omp parallel default (shared)
  {
    int i;

#pragma omp for
    for (i=0;  i<n;  i++) {
#pragma omp single
      work (i);
    }
  }


  for (j=0;  j<N;  j++) {
    if (x[j] != j) {
      errors += 1;
    }
  }

  if (errors == 0) {
    printf ("incorrect nest - PASSED\n");
  } else {
    printf ("incorrect nest - FAILED\n");
  }
}
