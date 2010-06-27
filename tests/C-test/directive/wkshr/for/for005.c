static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* for 005:
 * guided スケジューリングを指定した場合の動作を確認
 */

#include <omp.h>
#include "omni.h"


#define	MAX_CHUNK	10
#define LOOPNUM		(MAX_CHUNK*thds)

int ret_same (int);

int	thds;
int	*buf;


void
clear ()
{
  int lp;
  
  for (lp=0; lp<=LOOPNUM; lp++) {
    buf[lp] = -1;
  }
}


int
check_result (int s)
{
  int	lp, lp2, id;

  int	err = 0;


  for (lp=0; lp<LOOPNUM; ) {
    id = buf[lp];
    if (id<0 || thds<=id) {
      err += 1;
    }
    for (lp2=0; lp2<(LOOPNUM-lp); lp2++) {
      if (buf[lp+lp2] != id) {
	if (lp2+1<s) {
	  err += 1;
	}
      }
    }
    lp += lp2;
  }
  if (buf[LOOPNUM] != -1) {
    err += 1;
  }

  return err;
}


main ()
{
  int	lp;

  int	chunk = 3;
  int	errors = 0;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  buf = (int *) malloc (sizeof (int) * (LOOPNUM + 1));
  if (buf == NULL) {
    printf ("can not allocate memory.\n");
    exit (1);
  }
  omp_set_dynamic (0);

  clear ();
  #pragma omp parallel
  {
    #pragma omp for schedule (guided)
    for (lp=0; lp<LOOPNUM; lp++) {
      buf[lp] = omp_get_thread_num ();
    }
  }
  errors += check_result (1);

  clear ();
  #pragma omp parallel
  {
    #pragma omp for schedule (guided,1)
    for (lp=0; lp<LOOPNUM; lp++) {
      buf[lp] = omp_get_thread_num ();
    }
  }
  errors += check_result (1);

  clear ();
  #pragma omp parallel
  {
    #pragma omp for schedule (guided,2)
    for (lp=0; lp<LOOPNUM; lp++) {
      if (lp != 0) {
	while (buf[lp-1] == -1) {
	  waittime(1); /* for SCASH */
	  #pragma omp flush
	}
      }
      buf[lp] = omp_get_thread_num ();
      #pragma omp flush
    }
  }
  errors += check_result (2);

  clear ();
  #pragma omp parallel
  {
    #pragma omp for schedule (guided,chunk)
    for (lp=0; lp<LOOPNUM; lp++) {
      if (lp != 0) {
	while (buf[lp-1] == -1) {
	  waittime(1); /* for SCASH */
	  #pragma omp flush
	}
      }
      buf[lp] = omp_get_thread_num ();
      #pragma omp flush
    }
  }
  errors += check_result (chunk);

  clear ();
  #pragma omp parallel
  {
    #pragma omp for schedule (guided,2+1)
    for (lp=0; lp<LOOPNUM; lp++) {
      if (lp != 0) {
	while (buf[lp-1] == -1) {
	  waittime(1); /* for SCASH */
	  #pragma omp flush
	}
      }
      buf[lp] = omp_get_thread_num ();
      #pragma omp flush
    }
  }
  errors += check_result (2+1);

  clear ();
  #pragma omp parallel
  {
    #pragma omp for schedule (guided,ret_same(3))
    for (lp=0; lp<LOOPNUM; lp++) {
      if (lp != 0) {
	while (buf[lp-1] == -1) {
	  waittime(1); /* for SCASH */
	  #pragma omp flush
	}
      }
      buf[lp] = omp_get_thread_num ();
      #pragma omp flush
    }
  }
  errors += check_result (ret_same(3));

  clear ();
  #pragma omp parallel
  {
    #pragma omp for schedule (guided,ret_same(3)+1)
    for (lp=0; lp<LOOPNUM; lp++) {
      if (lp != 0) {
	while (buf[lp-1] == -1) {
	  waittime(1); /* for SCASH */
	  #pragma omp flush
	}
      }
      buf[lp] = omp_get_thread_num ();
      #pragma omp flush
    }
  }
  errors += check_result (ret_same(3)+1);

  clear ();
  #pragma omp parallel
  {
    int	ln = LOOPNUM;

    #pragma omp for schedule (guided,ln+1)
    for (lp=0; lp<LOOPNUM; lp++) {
      if (lp != 0) {
	while (buf[lp-1] == -1) {
	  waittime(1); /* for SCASH */
	  #pragma omp flush
	}
      }
      buf[lp] = omp_get_thread_num ();
      #pragma omp flush
    }
  }
  errors += check_result (LOOPNUM+1);

  if (errors == 0) {
    printf ("for 005 : SUCCESS\n");
    return 0;
  } else {
    printf ("for 005 : FAILED\n");
    return 1;
  }
}


int
ret_same (int i)
{
  return i;
};
