#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include "xmp.h"
#pragma xmp nodes p(*)
#define Mmin(a, b) (((a) < (b)) ? (a) : (b))
#define NTIMES 10
static int array_elements;
static double *restrict a, *restrict b, *restrict c;

static void checkSTREAMresults(const int num_nodes)
{
  double aj=1.0, bj=2.0, cj=0.0, scalar=3.0;
  double aSumErr=0.0, bSumErr=0.0, cSumErr=0.0, epsilon=1.e-13;

  /* Set valid value for only Triad */
  aj = bj + scalar * cj;

  /* Accumulate deltas between observed and expected results */
  for(int j=0;j<array_elements;j++){
    aSumErr += fabs(a[j] - aj);
    bSumErr += fabs(b[j] - bj);
    cSumErr += fabs(c[j] - cj);
  }
#pragma xmp reduction(+:aSumErr, bSumErr, cSumErr)
  double aAvgErr = aSumErr / (double)num_nodes / array_elements;
  double bAvgErr = bSumErr / (double)num_nodes / array_elements;
  double cAvgErr = cSumErr / (double)num_nodes / array_elements;

#pragma xmp task on p(1)
  {
    if(fabs(aAvgErr/aj) > epsilon || fabs(bAvgErr/bj) > epsilon || fabs(cAvgErr/cj) > epsilon)
      printf("Failed Validation %f %f %f\n", fabs(aAvgErr/aj), fabs(bAvgErr/bj), fabs(cAvgErr/cj));
    else
      printf("Solution Validates\n");
  }
}

static double HPCC_Stream()
{
  double scalar = 3.0, times[NTIMES], curGBs, mintime = FLT_MAX;
  int j, k;

  for(k=0;k<NTIMES;k++) {
#pragma xmp barrier
    times[k] = -xmp_wtime();

#ifdef _FUJITSU
#pragma loop xfill
#pragma loop noalias
#endif
#pragma omp parallel for
    for(j=0;j<array_elements;j++)
      a[j] = b[j] + scalar*c[j];

#pragma xmp barrier
    times[k] += xmp_wtime();
  }

  for(k=1;k<NTIMES;k++)
    mintime = Mmin(mintime, times[k]);

  curGBs = (mintime > 0.0 ? 1.0 / mintime : -1.0);
  curGBs *= 1e-9 * 3 * sizeof(double) * array_elements;
#pragma xmp reduction(+:curGBs)

  return curGBs;
}

int main(int argc, char **argv)
{
  int num_nodes = xmp_num_nodes();

  /* Set parameters */
  if(argc != 2){
#pragma xmp task on p(1)
    fprintf(stderr, "./STREAM (number of vector).\ne.g../STREAM 1000\n");
    return 1;
  }
  array_elements = atoi(argv[1]);

  /* Malloc arrays */
  a = malloc(sizeof(double) * array_elements);
  b = malloc(sizeof(double) * array_elements);
  c = malloc(sizeof(double) * array_elements);

  /* Initialize arrays */
#pragma omp parallel for
  for(int j=0;j<array_elements;j++){
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 0.0;
  }

  /* Execute STREAM */
  double triadGBs = HPCC_Stream();

#pragma xmp task on p(1)
  printf("[Vector size is %d] Total Triad %.4f GB/s on %d nodes\n", 
	 array_elements, triadGBs, num_nodes);

#include <omp.h>
#pragma xmp task on p(1)
  {
#pragma omp parallel
    {
#pragma omp single
      printf("Number of Threads requested = %d\n", omp_get_num_threads());
    }
  }
  
  /* Verification */
  checkSTREAMresults(num_nodes);

  return 0;
}
