static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* atomic 010:
 * double変数に対する、atomic の動作確認
 */

#include <omp.h>
#include "omni.h"


#define LOOPNUM		(thds * 2)

int		errors = 0;
int		thds;
double		atom_incr, atom_incr2, atom_decr, atom_decr2;
double		atom_plus, atom_mul, atom_minus, atom_div;


int
sameas (int v)
{
  return v;
}


void
clear ()
{
  atom_incr = 0;
  atom_incr2 = 0;
  atom_decr = 0;
  atom_decr2 = 0;

  atom_plus = 0;
  atom_minus = 0;
  atom_mul = 1;
  atom_div = 1;
}


int
check ()
{
  int		i;
  double	tmp;

  int	err = 0;


  if (atom_incr != LOOPNUM) {
    err ++;
  }
  if (atom_incr2 != LOOPNUM) {
    err ++;
  }
  if (atom_decr != -LOOPNUM) {
    err ++;
  }
  if (atom_decr2 != -LOOPNUM) {
    err ++;
  }

  if (atom_plus != LOOPNUM) {
    err ++;
  }
  if (atom_minus != -LOOPNUM) {
    err ++;
  }

  for (i=0, tmp=1; i<LOOPNUM; i++) {
    tmp = tmp * 2;
  }
  if (atom_mul != tmp) {
    err ++;
  }

  for (i=0, tmp=1; i<LOOPNUM; i++) {
    tmp = tmp / 2;
  }
  if (atom_div != tmp) {
    err ++;
  }

  return err;
}


main ()
{
  int	i;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  if (4 < thds) {
    thds = 4;
    omp_set_num_threads (4);
  }

  omp_set_dynamic (0);


  clear ();
  #pragma omp parallel
  {
    #pragma omp for 
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic
      atom_incr ++;
    }

    #pragma omp for 
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic
      ++ atom_incr2;
    }

    #pragma omp for 
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic
      atom_decr --;
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic
      -- atom_decr2;
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_plus += sameas(2) - 1;
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_minus -= sameas(2) - 1;
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_mul *= sameas(3) - 1;
    }

    #pragma omp for
    for (i=0; i<LOOPNUM; i++) {
      #pragma omp atomic	
      atom_div /= 4 + sameas(-2);
    }
  }
  errors += check ();

  if (errors == 0) {
    printf ("atomic 010 : SUCCESS\n");
    return 0;
  } else {
    printf ("atomic 010 : FAILED\n");
    return 1;
  }
}
