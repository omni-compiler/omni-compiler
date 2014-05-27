static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* private 026 :
 * private 宣言した変数をfor directiveのchunkサイズに指定した場合の動作を確認
 */

#include <omp.h>
#include "omni.h"


#define	LOOPNUM	(100 * thds)

int	errors = 0;
int	thds, *buff;

int	prvt;


void
check(int s)
{
  int	i,j, id;

  for (i=0; i<LOOPNUM; i+=s) {
    id = (i/s) % thds;

    for (j=0; j<s; j++) {
      if ((i+j) < LOOPNUM) {
	if (buff[i+j] != id) {
	  #pragma omp critical
	  errors += 1;
	}
      }
    }
  }
}


main ()
{
  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  buff = (int *)malloc(sizeof(int) * LOOPNUM);
  if (buff == NULL) {
    printf ("can not allocate memory.\n");
  }

  omp_set_dynamic (0);


  #pragma omp parallel private(prvt) 
  {
    int	i;

    for (prvt=1; prvt<=10; prvt++) {
      #pragma omp for schedule(static, prvt)
      for (i=0; i<LOOPNUM; i++) {
	buff[i] = omp_get_thread_num ();
      }
      check (prvt);

      #pragma omp barrier
    }
  }


  if (errors == 0) {
    printf ("private 026 : SUCCESS\n");
    return 0;
  } else {
    printf ("private 026 : FAILED\n");
    return 1;
  }
}
