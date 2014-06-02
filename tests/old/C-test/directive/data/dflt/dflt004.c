static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* default 004 :
 * default(none) が宣言されていている場合
 */

#include <omp.h>
#include "omni.h"


#define	MAGICNO	100


int	errors = 0;
int	thds;

int		shrd;


main ()
{
  int	i, t;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  t = 0;
  for (i=0; i<MAGICNO; i++) {
    t += i;
  }


  #pragma omp parallel default(none) shared(shrd)
  {
    #pragma omp for
    for (i=0; i<MAGICNO; i++) {
      #pragma omp critical
      shrd += i;
    }
  }
  if (shrd != t) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("default 004 : SUCCESS\n");
    return 0;
  } else {
    printf ("default 004 : FAILED\n");
    return 1;
  }
}
