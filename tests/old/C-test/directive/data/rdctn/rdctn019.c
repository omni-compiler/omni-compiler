static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* reduction 019 :
 * double 変数に reduction が指定された場合の動作確認
 */

#include <omp.h>
#include "omni.h"


#define	LOOPNUM	(thds * 2)

int	errors = 0;
int	thds;

double	rdct_inc, rdct_inc2, rdct_pls, rdct_pls2, rdct_pls3;
double	rdct_dec, rdct_dec2, rdct_mns, rdct_mns2;
double	rdct_mul, rdct_mul2, rdct_mul3;
double	rdct_and, rdct_and2;
double	rdct_or, rdct_or2;

double	rst_inc, rst_inc2, rst_pls, rst_pls2, rst_pls3;
double	rst_dec, rst_dec2, rst_mns, rst_mns2;
double	rst_mul, rst_mul2, rst_mul3;
double	rst_and, rst_and2;
double	rst_or, rst_or2;


main ()
{
  int	i;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  rdct_inc = rdct_inc2 = rdct_pls = rdct_pls2 = rdct_pls3 = 1;
  rdct_dec = rdct_dec2 = rdct_mns = rdct_mns2 = 2;
  rdct_mul = rdct_mul2 = rdct_mul3 = 3;
  rdct_and = rdct_and2 = 1;
  rdct_or = rdct_or2 = 0;

  #pragma omp parallel for reduction(+:rdct_inc,rdct_inc2,rdct_pls,rdct_pls2,rdct_pls3) \
			   reduction(-:rdct_dec,rdct_dec2,rdct_mns,rdct_mns2) \
			   reduction(*:rdct_mul,rdct_mul2,rdct_mul3) \
			   reduction (&&:rdct_and,rdct_and2) \
			   reduction (||:rdct_or,rdct_or2)
  for (i=0; i<LOOPNUM; i++) {

    rdct_inc ++;
    ++ rdct_inc2;
    rdct_pls += i;
    rdct_pls2 = rdct_pls2 + i;
    rdct_pls3 = i + rdct_pls3;

    rdct_dec --;
    -- rdct_dec2;
    rdct_mns -= i;
    rdct_mns2 = rdct_mns2 - i;

    rdct_mul *= i;
    rdct_mul2 = rdct_mul2 * i;
    rdct_mul3 = i * rdct_mul3;

    rdct_and = rdct_and && i;
    rdct_and2 = (i+1) && rdct_and2;

    rdct_or = rdct_or || i;
    rdct_or2 = 0 || rdct_or2;

    if (sizeof(rdct_inc) != sizeof(double)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_inc2) != sizeof(double)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_pls) != sizeof(double)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_pls2) != sizeof(double)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_pls3) != sizeof(double)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_dec) != sizeof(double)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_dec2) != sizeof(double)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_mns) != sizeof(double)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_mns2) != sizeof(double)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_mul) != sizeof(double)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_mul2) != sizeof(double)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_mul3) != sizeof(double)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_and) != sizeof(double)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_and2) != sizeof(double)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_or) != sizeof(double)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_or2) != sizeof(double)) {
      #pragma omp critical
      errors += 1;
    }
  }


  rst_inc = rst_inc2 = rst_pls = rst_pls2 = rst_pls3 = 1;
  rst_dec = rst_dec2 = rst_mns = rst_mns2 = 2;
  rst_mul = rst_mul2 = rst_mul3 = 3;
  rst_and = rst_and2 = 1;
  rst_or = rst_or2 = 0;

  for (i=0; i<LOOPNUM; i++) {

    rst_inc ++;
    ++ rst_inc2;
    rst_pls += i;
    rst_pls2 = rst_pls2 + i;
    rst_pls3 = i + rst_pls3;

    rst_dec --;
    -- rst_dec2;
    rst_mns -= i;
    rst_mns2 = rst_mns2 - i;

    rst_mul *= i;
    rst_mul2 = rst_mul2 * i;
    rst_mul3 = i * rst_mul3;

    rst_and = rst_and && i;
    rst_and2 = (i+1) && rst_and2;

    rst_or = rst_or || i;
    rst_or2 = 0 || rst_or2;
  }

  if (rst_inc != rdct_inc) {
    errors += 1;
  }
  if (rst_inc2 != rdct_inc2) {
    errors += 1;
  }
  if (rst_pls != rdct_pls) {
    errors += 1;
  }
  if (rst_pls2 != rdct_pls2) {
    errors += 1;
  }
  if (rst_pls3 != rdct_pls3) {
    errors += 1;
  }
  if (rst_dec != rdct_dec) {
    errors += 1;
  }
  if (rst_dec2 != rdct_dec2) {
    errors += 1;
  }
  if (rst_mns != rdct_mns) {
    errors += 1;
  }
  if (rst_mns2 != rdct_mns2) {
    errors += 1;
  }
  if (rst_mul != rdct_mul) {
    errors += 1;
  }
  if (rst_mul2 != rdct_mul2) {
    errors += 1;
  }
  if (rst_mul3 != rdct_mul3) {
    errors += 1;
  }
  if (rst_and != rdct_and) {
    errors += 1;
  }
  if (rst_and2 != rdct_and2) {
    errors += 1;
  }
  if (rst_or != rdct_or) {
    errors += 1;
  }
  if (rst_or2 != rdct_or2) {
    errors += 1;
  }


  if (errors == 0) {
    printf ("reduction 019 : SUCCESS\n");
    return 0;
  } else {
    printf ("reduction 019 : FAILED\n");
    return 1;
  }
}
