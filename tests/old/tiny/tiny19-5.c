/* incorrect case
 * The following example is incorrect bacause the barrier results in
 * deadlock due to the fact that only one thread executes the single section
 */

#include <stdio.h>

#define N       1024

int     x;

work (i)
     int        i;
{
  x = i;
}


main ()
{
  int   errors = 0;


  printf ("!!! this program is wrong.\n"
          "!!! always FAILED, dead lock or compile error\n");

#pragma omp parallel
  {
#pragma omp single
    {
      work (0);
#pragma omp barrier
      work (1);
    }
  }

  if (x != 1) {
    errors += 1;
  }

  if (errors == 0) {
    printf ("incorrect nest - PASSED\n");
  } else {
    printf ("incorrect nest - FAILED\n");
  }
}
