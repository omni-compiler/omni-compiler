/* incorrect case
 * OpenMP C API Test Suite
 * Example A.19 from OpenMP C/C++ API sepecification
 */
#include <stdio.h>

#define N       1024

int     x[N][N];

work2 (i,j)
     int        i, j;
{
  x[i][j] = i*j;
}


work1 (i, n)
     int        i, n;
{
  int   j;

#pragma omp for
  for (j=0;  j<n;  j++) {
    work2 (i, j);
  }
}


main ()
{
  int   i, j;

  int   n = N;
  int   errors = 0;


  printf ("!!! this program is wrong.\n"
          "!!! always FAILED or compile error\n");

  for (i=0;  i<N;  i++) {
    for (j=0;  j<N;  j++) {
      x[i][j] = -1;
    }
  }

#pragma omp parallel default (shared)
  {
#pragma omp for
    for (i=0;  i<n;  i++) {
      work1 (i, n);
    }
  }

  for (i=0;  i<N;  i++) {
    for (j=0;  j<N;  j++) {
      if (x[i][j] != i*j) {
        errors += 1;
      }
    }
  }

  if (errors == 0) {
    printf ("incorrect nest - PASSED\n");
  } else {
    printf ("incorrect nest - FAILED\n");
  }
}
