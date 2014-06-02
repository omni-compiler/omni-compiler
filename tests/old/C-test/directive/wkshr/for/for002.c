static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* for 002:
 * canonical loop の動作確認
 * canonical shape is :
 *   for (init-expr; var logical-op b; incr-expr)
 *   init-expr = var = lb
 *   incr-expr = ++var,
 *		 var++,
 *		 --var,
 *		 var--,
 *		 var += incr,
 *		 var -= incr,
 *		 var = var + incr,
 *		 var = incr + var,
 *		 var = var - incr,
 *   logical-op = <, <=, >, >=
 */

#include <omp.h>
#include "omni.h"


int	thds, iter;
int	*buf;


void clear ()
{
  int lp;

  for (lp=0; lp<iter; lp++) {
    buf[lp] = 0;
  }
  buf[iter] = -1;
}


int
check_result (int v)
{
  int	lp;

  int	err = 0;


  for (lp = 0; lp<iter; lp++) {
    if (buf[lp] != lp % v) {
      err += 1;
    }
  }
  if (buf[iter] != -1) {
    err += 1;
  }

  return err;
}


void
test_for_001 ()
{
  int	lp;

  #pragma omp for schedule(static,1)
  for (lp=0; lp<iter; lp++) {
    buf[lp] = omp_get_thread_num ();
  }
}

void
test_for_002 ()
{
  int	lp;
  int	var = 0;

  #pragma omp for schedule(static,1)
  for (lp=var; lp<iter; lp++) {
    buf[lp] += omp_get_thread_num ();
  }
}

void
test_for_003 ()
{
  int	lp;
  int	var = 10;

  #pragma omp for schedule(static,1)
  for (lp=var/10-1; lp<iter; lp++) {
    buf[lp] += omp_get_thread_num ();
  }
}

void
test_for_004 ()
{
  int	lp;

  #pragma omp for schedule(static,1)
  for (lp=0; lp<iter; lp++) {
    buf[lp] += omp_get_thread_num ();
  }
}

void
test_for_005 ()
{
  int	lp;

  #pragma omp for schedule(static,1)
  for (lp=1; lp<=iter; lp++) {
    buf[lp-1] += omp_get_thread_num ();
  }
}

void
test_for_006 ()
{
  int	lp;

  #pragma omp for schedule(static,1)
  for (lp=0; lp>-iter; lp--) {
    buf[-lp] += omp_get_thread_num ();
  }
}

void
test_for_007 ()
{
  int	lp;

  #pragma omp for schedule(static,1)
  for (lp=-1; lp>=-iter; lp--) {
    buf[-(lp+1)] += omp_get_thread_num ();
  }
}

void
test_for_010 ()
{
  int	lp, incr = 1;

  #pragma omp for schedule(static,1)
  for (lp=0; lp<iter; lp += incr) {
    buf[lp] += omp_get_thread_num ();
  }
}

void
test_for_011 ()
{
  int	lp, incr = 1;

  #pragma omp for schedule(static,1)
  for (lp=0; lp<iter; lp = lp + incr) {
    buf[lp] += omp_get_thread_num ();
  }
}

void
test_for_012 ()
{
  int	lp, incr = 1;

  #pragma omp for schedule(static,1)
  for (lp=0; lp<iter; lp = incr + lp) {
    buf[lp] += omp_get_thread_num ();
  }
}

void
test_for_013 ()
{
  int	lp;

  #pragma omp for schedule(static,1)
  for (lp=0; lp>-iter; lp--) {
    buf[-lp] += omp_get_thread_num ();
  }
}

void
test_for_014 ()
{
  int	lp;

  #pragma omp for schedule(static,1)
  for (lp=0; lp>-iter; --lp) {
    buf[-lp] += omp_get_thread_num ();
  }
}

void
test_for_015 ()
{
  int	lp, decr = 1;

  #pragma omp for schedule(static,1)
  for (lp=0; lp>-iter; lp -= decr) {
    buf[-lp] += omp_get_thread_num ();
  }
}

void
test_for_016 ()
{
  int	lp, decr = 1;

  #pragma omp for schedule(static,1)
  for (lp=0; lp>-iter; lp = lp - decr) {
    buf[-lp] += omp_get_thread_num ();
  }
}

void
test_for_017 ()
{
  int	lp, decr = 1;

  #pragma omp for schedule(static,1)
  for (lp=0; lp>-iter; lp = - decr + lp) {
    buf[-lp] += omp_get_thread_num ();
  }
}

void
test_for_018 ()
{
  int	lp;

  #pragma omp for schedule(static,1)
  for (lp=0; lp>-iter; --lp) {
    buf[-lp] += omp_get_thread_num ();
  }
}

void
test_for_019 ()
{
  int	lp;

  #pragma omp for schedule(static,1)
  for (lp=0; lp<iter; ++lp) {
    buf[lp] += omp_get_thread_num ();
  }
}

void
test_for_020 ()
{
  int	lp, decr = 1, incr = 1;

  #pragma omp for schedule(static,1)
  for (lp=0; lp<iter; lp = 2*incr - decr + lp) {
    buf[lp] += omp_get_thread_num ();
  }
}


void
test_for_021 ()
{
  int	lp, decr = 1, incr = 1;

  #pragma omp for schedule(static,1)
  for (lp=0; lp<iter; lp = lp + (2*incr - decr)) {
    buf[lp] += omp_get_thread_num ();
  }
}

void
test_for_022 ()
{
  int	lp, decr = 1, incr = 1;

  #pragma omp for schedule(static,1)
  for (lp=0; lp<iter+decr-incr; lp++) {
    buf[lp] += omp_get_thread_num ();
  }
}

void
test_for_023 ()
{
  int	lp, decr = 1, incr = 1;

  #pragma omp for schedule(static,1)
  for (lp=0; lp<iter; lp+=incr*2-decr) {
    buf[lp] += omp_get_thread_num ();
  }
}


main ()
{
  int	errors = 0;

  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  iter = thds * 2;
  buf = (int *) malloc (sizeof (int) * (iter + 1));
  if (buf == NULL) {
    printf ("can not allocate memory.\n");
    exit (1);
  }
  omp_set_dynamic (0);

  {
    int	buf[11], i;
    for (i=0; i<11; i++) {
      buf[i] = -1;
    }
    #pragma omp parallel
    {
      #pragma omp for schedule(static,1)
      for (i=0; i<10; i++) {
	buf[i] = omp_get_thread_num ();
      }
    }
    for (i=0; i<10; i++) {
      if (buf[i] != i % thds) {
	errors += 1;
      }
    }
    if (buf[10] != -1) {
      errors += 1;
    }
  } 

  clear ();
  #pragma omp parallel
  test_for_001 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_002 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_003 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_004 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_005 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_006 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_007 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_010 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_011 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_012 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_013 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_014 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_015 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_016 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_017 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_018 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_019 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_020 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_021 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_022 ();
  errors += check_result (thds);

  clear ();
  #pragma omp parallel
  test_for_023 ();
  errors += check_result (thds);

  if (errors == 0) {
    printf ("for 002 : SUCCESS\n");
    return 0;
  } else {
    printf ("for 002 : FAILED\n");
    return 1;
  }
}
