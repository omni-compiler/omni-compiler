static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* omp_test_lock 002:
 * スレッド間で、omp_test_nest_lock関数の動作を確認します。
 */

#include <signal.h>
#include <sys/time.h>
#include <omp.h>
#include "omni.h"


#if 0
void
set_timeout(int t, void (*func)(int))
{
  struct itimerval	it;


  signal (SIGALRM, func);

  it.it_interval.tv_sec = 0;
  it.it_interval.tv_usec = 0;
  it.it_value.tv_sec = t;
  it.it_value.tv_usec = 0;
  setitimer(ITIMER_REAL, &it, NULL);
}


void
deadlock (int sig)
{
  printf ("dead locked !!\n");
  printf ("omp_test_nest_lock 002 : FAILED\n");
  exit (1);
}
#endif


main ()
{
  omp_nest_lock_t	lck;
  int			thds, s, f;

  int			errors = 0;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi thread.\n");
    exit (0);
  }
  omp_set_dynamic (0);

  omp_init_nest_lock(&lck);
  s = f = 0;

#if 0
  set_timeout(10+thds, deadlock);
#endif

  #pragma omp parallel
  {
    int	t;

    #pragma omp barrier

    t = omp_test_nest_lock (&lck);
    if (t != 0) {
      /* nest_lock successful */
      barrier (thds);		      /* barrier 1 */
      t = omp_test_nest_lock (&lck);
      if (t != 2) {
	#pragma omp critical
	errors += 1;
      }
      omp_unset_nest_lock (&lck);
      barrier (thds);		      /* barrier 2 */
      barrier (thds);		      /* barrier 3 */
      omp_unset_nest_lock (&lck);
      barrier (thds);		      /* barrier 4 */
      barrier (thds);		      /* barrier 5 */

    } else {
      /* nest_lock fail */
      #pragma omp critical
      f += 1;

      barrier (thds);		      /* barrier 1 */
      t = omp_test_nest_lock (&lck);
      if (t != 0) {
	#pragma omp critical
	errors += 1;
      }
      barrier (thds);		      /* barrier 2 */
      t = omp_test_nest_lock (&lck);
      if (t != 0) {
	#pragma omp critical
	errors += 1;
      }
      barrier (thds);		      /* barrier 3 */
      barrier (thds);		      /* barrier 4 */
      t = omp_test_nest_lock (&lck);
      if (t == 1) {
	#pragma omp critical
	s += 1;
      } else if (t != 0) {
	#pragma omp critical
	errors += 1;
      }
      barrier (thds);		      /* barrier 5 */
    }
  }

  if (s != 1) {
    errors += 1;
  }
  if (f != thds - 1) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("omp_test_nest_lock 002 : SUCCESS\n");
    return 0;
  } else {
    printf ("omp_test_nest_lock 002 : FAILED\n");
    return 1;
  }
}
