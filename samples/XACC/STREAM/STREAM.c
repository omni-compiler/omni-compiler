#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <openacc.h>
#include "xmp.h"
#define Mmin(a, b) (((a) < (b)) ? (a) : (b))
#define NTIMES 10
#pragma xmp nodes p(*)

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

static double HPCC_Stream(const double ratio)
{
  int GPU_SIZE  = array_elements * ratio;
  double scalar = 3.0, times[NTIMES], curGBs, mintime = FLT_MAX;

#pragma acc data copy(a[0:GPU_SIZE], b[0:GPU_SIZE], c[0:GPU_SIZE])
  {
    for(int k=0;k<NTIMES;k++) {
#pragma xmp barrier
      times[k] = -xmp_wtime();

#pragma acc parallel loop async
      for(int j=0;j<GPU_SIZE;j++)
	a[j] = b[j] + scalar*c[j];

#pragma omp parallel for
      for(int j=GPU_SIZE;j<array_elements;j++)
	a[j] = b[j] + scalar*c[j];

#pragma acc wait

#pragma xmp barrier
      times[k] += xmp_wtime();
    }
  } // end acc data

  for(int k=1;k<NTIMES;k++)
    mintime = Mmin(mintime, times[k]);

  curGBs = (mintime > 0.0 ? 1.0 / mintime : -1.0);
  curGBs *= 1e-9 * 3 * sizeof(double) * array_elements;
#pragma xmp reduction(+:curGBs)

  return curGBs;
}

int main(int argc, char **argv)
{
  double GiB    = 1024.0*1024.0*1024.0;
  int num_nodes = xmp_num_nodes();

  /* Set parameters */
  if(argc != 3){
#pragma xmp task on p(1)
    fprintf(stderr, "./STREAM (number of vector) (ratio of GPU memory).\ne.g../STREAM 1000 0.5\n");
    return 1;
  }
  array_elements = atoi(argv[1]);
  long  allsize  = array_elements * sizeof(double) * 3;
  double ratio   = atof(argv[2]);

  /* Malloc arrays */
  a = malloc(sizeof(double) * array_elements);
  b = malloc(sizeof(double) * array_elements);
  c = malloc(sizeof(double) * array_elements);

#pragma xmp task on p(1)
  printf("Ratio %.2f: GPU %.2f GiB CPU %.2f GiB\n", ratio, allsize*ratio/GiB, allsize*(1-ratio)/GiB);

  /* Set different GPUs */
  int ngpus  = acc_get_num_devices(acc_device_nvidia);
  int gpunum = (xmp_node_num()-1) % ngpus + 1;
  acc_set_device_num(gpunum, acc_device_nvidia);

  /* Initialize arrays */
#pragma omp parallel for
  for(int j=0;j<array_elements;j++){
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 0.0;
  }

  /* Execute STREAM */
  double triadGBs = HPCC_Stream(ratio);

#pragma xmp task on p(1)
  printf("[Vector size is %d] Total Triad %.2f GB/s on %d nodes\n", 
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
