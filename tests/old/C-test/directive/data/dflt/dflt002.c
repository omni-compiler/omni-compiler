static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* default 002 :
 * default(shared) が宣言されていて、data attributeが宣言されている場合の動作確認
 */

#include <omp.h>
#include "omni.h"


#define	MAGICNO	100


int	errors = 0;
int	thds;

int	tprvt, prvt, fprvt, lprvt, rdct, shrd, shrd2;
#pragma omp threadprivate (tprvt)


const int	cnst = MAGICNO;


main ()
{
  int	i, r;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);

  rdct = shrd = 0;
  fprvt = MAGICNO;
  #pragma omp parallel for default(shared) private (prvt) firstprivate(fprvt) lastprivate(lprvt) reduction(+:rdct) shared(shrd) schedule (static,1)
  for (i=0; i<thds; i++) {
    #pragma omp critical
    {
      shrd  += 6*i;		      /* shrd is shared, i is private */
      shrd2 += 7*i;		      /* shrd is shared */
    }
    tprvt  = i;			      /* tprvt is threadprivate */
    prvt   = 2*i;		      /* prvt is private */
    fprvt += 3*i;		      /* fprvt is firstprivate */
    lprvt  = 4*i;		      /* lprvt is lastprivate */
    rdct  += 5*i;		      /* rdct is reduction(+) */
    waittime (1);

    if (prvt != 2*i) {		      /* check private */
      #pragma omp critical
      errors += 1;
    }
    if (fprvt != MAGICNO + 3*i) {
      #pragma omp critical
      errors += 1;
    }
  }

  r = 0;
  for (i=0; i<thds; i++) 
    r += i;
  if (rdct != r * 5) {
    errors += 1;
  }

  if (shrd != r * 6) {
    errors += 1;
  }
  if (shrd2 != r * 7) {
    errors += 1;
  }

  if (lprvt != 4*(thds-1)) {
    errors += 1;
  }

  #pragma omp parallel for default(shared) schedule (static)
  for (i=0; i<thds; i++) {
    if (tprvt != i) {
      #pragma omp critical
      errors += 1;
    }
  }


  if (errors == 0) {
    printf ("default 002 : SUCCESS\n");
    return 0;
  } else {
    printf ("default 002 : FAILED\n");
    return 1;
  }
}
