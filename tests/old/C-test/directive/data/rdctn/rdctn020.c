static char rcsid[] = "$Id$";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/* reduction 020 :
 * enum 変数に reduction が指定された場合の動作確認
 */

#include <omp.h>
#include "omni.h"


#define	LOOPNUM	(thds * 2)

int	errors = 0;
int	thds;

enum x { ZERO, ONE, TWO, THREE };

enum x	rdct_inc, rdct_inc2, rdct_pls, rdct_pls2, rdct_pls3;
enum x	rdct_dec, rdct_dec2, rdct_mns, rdct_mns2;
enum x	rdct_mul, rdct_mul2, rdct_mul3;
enum x	rdct_land, rdct_land2, rdct_land3;
enum x	rdct_lor, rdct_lor2, rdct_lor3;
enum x	rdct_xor, rdct_xor2, rdct_xor3;
enum x	rdct_and, rdct_and2;
enum x	rdct_or, rdct_or2;

enum x	rst_inc, rst_inc2, rst_pls, rst_pls2, rst_pls3;
enum x	rst_dec, rst_dec2, rst_mns, rst_mns2;
enum x	rst_mul, rst_mul2, rst_mul3;
enum x	rst_land, rst_land2, rst_land3;
enum x	rst_lor, rst_lor2, rst_lor3;
enum x	rst_xor, rst_xor2, rst_xor3;
enum x	rst_and, rst_and2;
enum x	rst_or, rst_or2;


main ()
{
  int	i;


  thds = omp_get_max_threads ();
  if (thds == 1) {
    printf ("should be run this program on multi threads.\n");
    exit (0);
  }
  omp_set_dynamic (0);


  rdct_inc = rdct_inc2 = rdct_pls = rdct_pls2 = rdct_pls3 = ONE;
  rdct_dec = rdct_dec2 = rdct_mns = rdct_mns2 = TWO;
  rdct_mul = rdct_mul2 = rdct_mul3 = THREE;
  rdct_land = rdct_land2 = rdct_land3 = (enum x)-2;
  rdct_lor = rdct_lor2 = rdct_lor3 = ONE;
  rdct_xor = rdct_xor2 = rdct_xor3 = TWO;
  rdct_and = rdct_and2 = ONE;
  rdct_or = rdct_or2 = ZERO;

  #pragma omp parallel for reduction(+:rdct_inc,rdct_inc2,rdct_pls,rdct_pls2,rdct_pls3) \
			   reduction(-:rdct_dec,rdct_dec2,rdct_mns,rdct_mns2) \
			   reduction(*:rdct_mul,rdct_mul2,rdct_mul3) \
			   reduction(&:rdct_land,rdct_land2,rdct_land3) \
			   reduction (|:rdct_lor,rdct_lor2,rdct_lor3) \
			   reduction (^:rdct_xor,rdct_xor2,rdct_xor3) \
			   reduction (&&:rdct_and,rdct_and2) \
			   reduction (||:rdct_or,rdct_or2)
  for (i=0; i<LOOPNUM; i++) {

    rdct_inc ++;
    ++ rdct_inc2;
    rdct_pls += (enum x)(i);
    rdct_pls2 = (enum x)(rdct_pls2 + i);
    rdct_pls3 = (enum x)(i + rdct_pls3);

    rdct_dec --;
    -- rdct_dec2;
    rdct_mns -= (enum x)(i);
    rdct_mns2 = (enum x)(rdct_mns2 - i);

    rdct_mul *= (enum x)(i);
    rdct_mul2 = (enum x)(rdct_mul2 * i);
    rdct_mul3 = (enum x)(i * rdct_mul3);

    rdct_land &= (enum x)(1<<i);
    rdct_land2 = (enum x)(rdct_land2 & (1<<i));
    rdct_land3 = (enum x)((1<<i) & rdct_land3);

    rdct_lor |= (enum x)(1<<i);
    rdct_lor2 = (enum x)(rdct_lor2 | (1<<i));
    rdct_lor3 = (enum x)((1<<i) | rdct_lor3);

    rdct_xor ^= (enum x)(1<<i);
    rdct_xor2 = (enum x)(rdct_xor2 ^ (1<<i));
    rdct_xor3 = (enum x)((1<<i) ^ rdct_xor3);

    rdct_and = (enum x)(rdct_and && i);
    rdct_and2 = (enum x)((i+1) && rdct_and2);

    rdct_or = (enum x)(rdct_or || i);
    rdct_or2 = (enum x)(0 || rdct_or2);

    if (sizeof(rdct_inc) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_inc2) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_pls) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_pls2) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_pls3) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_dec) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_dec2) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_mns) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_mns2) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_mul) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_mul2) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_mul3) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_land) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_land2) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_land3) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_lor) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_lor2) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_lor3) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_xor) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_xor2) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_xor3) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_and) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_and2) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_or) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
    if (sizeof(rdct_or2) != sizeof(enum x)) {
      #pragma omp critical
      errors += 1;
    }
  }

  rst_inc = rst_inc2 = rst_pls = rst_pls2 = rst_pls3 = ONE;
  rst_dec = rst_dec2 = rst_mns = rst_mns2 = TWO;
  rst_mul = rst_mul2 = rst_mul3 = THREE;
  rst_land = rst_land2 = rst_land3 = (enum x)-2;
  rst_lor = rst_lor2 = rst_lor3 = ONE;
  rst_xor = rst_xor2 = rst_xor3 = TWO;
  rst_and = rst_and2 = ONE;
  rst_or = rst_or2 = ZERO;

  for (i=0; i<LOOPNUM; i++) {

    rst_inc ++;
    ++ rst_inc2;
    rst_pls += (enum x)(i);
    rst_pls2 = (enum x)(rst_pls2 + i);
    rst_pls3 = (enum x)(i + rst_pls3);

    rst_dec --;
    -- rst_dec2;
    rst_mns -= (enum x)(i);
    rst_mns2 = (enum x)(rst_mns2 - i);

    rst_mul *= (enum x)(i);
    rst_mul2 = (enum x)(rst_mul2 * i);
    rst_mul3 = (enum x)(i * rst_mul3);

    rst_land &= (enum x)(1<<i);
    rst_land2 = (enum x)((int)rst_land2 & (1<<i));
    rst_land3 = (enum x)((1<<i) & rst_land3);

    rst_lor |= (enum x)(1<<i);
    rst_lor2 = (enum x)(rst_lor2 | (1<<i));
    rst_lor3 = (enum x)((1<<i) | rst_lor3);

    rst_xor ^= (enum x)(1<<i);
    rst_xor2 = (enum x)((int)rst_xor2 ^ (1<<i));
    rst_xor3 = (enum x)((1<<i) ^ rst_xor3);

    rst_and = (enum x)(rst_and && i);
    rst_and2 = (enum x)((i+1) && rst_and2);

    rst_or = (enum x)(rst_or || i);
    rst_or2 = (enum x)(0 || rst_or2);
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
  if (rst_land != rdct_land) {
    errors += 1;
  }
  if (rst_land2 != rdct_land2) {
    errors += 1;
  }
  if (rst_land3 != rdct_land3) {
    errors += 1;
  }
  if (rst_lor != rdct_lor) {
    errors += 1;
  }
  if (rst_lor2 != rdct_lor2) {
    errors += 1;
  }
  if (rst_lor3 != rdct_lor3) {
    errors += 1;
  }
  if (rst_xor != rdct_xor) {
    errors += 1;
  }
  if (rst_xor2 != rdct_xor2) {
    errors += 1;
  }
  if (rst_xor3 != rdct_xor3) {
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
    printf ("reduction 020 : SUCCESS\n");
    return 0;
  } else {
    printf ("reduction 020 : FAILED\n");
    return 1;
  }
}
