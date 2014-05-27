
#ifndef __OMNI_CTEST_H__
#define __OMNI_CTEST_H__

#include "config.h"

#ifdef __OMNI_SCASH__
#include <omp.h>
#define malloc(X)	ompsm_galloc(X,OMNI_DEST_NONE,0)
#define waittime(X)	{ int x; _ompsm_scash_lib_in(); for(x=0; x<(X)*10000; x++) scash_poll(); _ompsm_scash_lib_out(); }

#else /* __OMNI_SCASH__ */
#ifdef __OMNI_SHMEM__
#include <omp.h>
#define malloc(X)	ompsm_galloc(X,OMNI_DEST_NONE,0)
#define waittime(X)	sleep(X)

#else /* __OMNI_SHMEM__ */
#define waittime(X)	sleep(X)
#endif /* __OMNI_SHMEM__ */
#endif /* __OMNI_SCASH__ */


#include <stdio.h>
#if defined(__OMNI_SCASH__) || defined(__OMNI_SHMEM__)
#pragma omp threadprivate (stdin,stdout,stderr)
#endif /* defined(__OMNI_SCASH__) || defined(__OMNI_SHMEM__) */

#define MAXERROR	10
#define ERROR(error_count)		                    	\
  {                                                             \
    error_count ++;                                             \
    if (error_count <= MAXERROR) {				\
      fprintf(stderr, "error detected at line %d\n", __LINE__);	\
    }								\
  }								\


static void
barrier (int n)
{
  static int cnt[2] = { 0, 0 };
  static int cur = 0;

  int	c;

  #pragma omp critical
  {
    c = cur;
    if (cnt[c]+1 == n) {
      cur = (cur + 1) % 2;
      cnt[c] = 0;
    } else {
      cnt[c] += 1;
    }
  }

  while (cnt[c] != 0) {
    waittime(1);
    #pragma omp flush
  }
}

void exit(int);


#endif /* __OMNI_CTEST_H__ */
