#include "xmp.h"
#include <stdlib.h>
#include <stdio.h>

void xmp_sort_up(xmp_desc_t a, xmp_desc_t b);

int main(int argc, char *argv[]){

#pragma xmp nodes p(4)

  int me = xmp_node_num() - 1;
  int nprocs = xmp_num_nodes();

  int m[nprocs];
  int n = 0;

  for (int i = 0; i < nprocs; i++){
    m[i] = (i + 1) * nprocs * 100;
    n += m[i];
  }

  int dummy;

#pragma xmp template t(0:n-1)
#pragma xmp distribute t(gblock(m)) onto p

  //
  // int
  //

  if (me == 0) printf("check for int starts...\n");

  int a0[n], b0[n];
#pragma xmp align a0[i] with t(i)
#pragma xmp align b0[i] with t(i)
#pragma xmp shadow a0[1:1]
#pragma xmp shadow b0[1:1]

  srand(me);

#pragma xmp loop on t(i)
  for (int i = 0; i < n; i++){
    a0[i] = rand() % n;
  }

  xmp_sort_up(xmp_desc_of(a0), xmp_desc_of(b0));

  //
  // float
  //

  if (me == 0) printf("check for float starts...\n");

  float a1[n], b1[n];
#pragma xmp align a1[i] with t(i)
#pragma xmp align b1[i] with t(i)
#pragma xmp shadow a1[1:1]
#pragma xmp shadow b1[1:1]

  srand(me);

#pragma xmp loop on t(i)
  for (int i = 0; i < n; i++){
    a1[i] = ((float)rand() / ((float)RAND_MAX + 1)) * 100;
  }

  xmp_sort_up(xmp_desc_of(a1), xmp_desc_of(b1));

}
