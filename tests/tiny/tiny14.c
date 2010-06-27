/*
 * OpenMP C API Test Suite
 * Example A.14 from OpenMP C/C++ API sepecification
 */
int omp_get_max_threads(void);

#include <stdio.h>


int     x;
int     *p;


void
f1(q)
     int        *q;
{
  *q = 1;
#pragma omp flush
  /* x, p, and *q are flushed
   * because they are shared and accessible 
   */
}


void
f2(q)
     int        *q;
{
  *q = 2;
#pragma omp barrier
  /* a barrier implies a flush
   * x, p, and *q are flushed
   * because they are shared and accessible
   */
}


int
g(n)
     int        n;
{
  int   j, k, tmp;

  int   i = 1, sum = 0;


  *p = 1;
#pragma omp parallel reduction(+:sum)
  {
    f1 (&j);
    /* i and n were not flushed
     * because they were not accessible in f1
     * j was flushed because it was accessible
     */
    sum += j;

#pragma omp barrier
    f2(&j);
    /* i and n were not flushed
     * because they were not acessible in f2
     * j was flushed because it was accessible
     */
    sum += i + j + *p + n;
  }
  return sum;
}


main ()
{
  int   ret, thds;

  int   errors = 0;

  p = &x;

  ret = g (10);

  thds = omp_get_max_threads ();
  if (ret != 15*thds) {
    errors += 1;
    printf ("flush 002 - expected sum = %d, observed %d\n",
            ret, 15 * thds);
  }

  if (errors == 0) {
    printf ("flush 002 PASSED\n");
  } else {
    printf ("flush 002 FAILED\n");
  }
}

