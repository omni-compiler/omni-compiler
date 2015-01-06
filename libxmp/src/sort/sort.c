#include "xmp.h"
#include <stdlib.h>
#include <stdio.h>

void xmp_sort_up(xmp_desc_t a, xmp_desc_t b);
void xmp_sort_down(xmp_desc_t a, xmp_desc_t b);

int main(int argc, char *argv[]){

#pragma xmp nodes p(*)

  int me = xmp_node_num() - 1;
  int nprocs = xmp_num_nodes();

  int m[nprocs];
  int n = 0;

  for (int i = 0; i < nprocs; i++){
    m[i] = (i + 1) * nprocs * 5;
    n += m[i];
  }

  int result;

#pragma xmp template t0(0:n-1)
#pragma xmp distribute t0(gblock(m)) onto p

#pragma xmp template t1(0:n-1)
#pragma xmp distribute t1(block) onto p

#pragma xmp template t2(0:n-1)
#pragma xmp distribute t2(cyclic) onto p

  //#pragma xmp distribute t(block) onto p

  //
  // int
  //

  //if (me == 0) printf("check for int starts...\n");

  int a0[n], b0[n];
#pragma xmp align a0[i] with t0(i)
#pragma xmp align b0[i] with t1(i)
#pragma xmp shadow b0[1:1]

  srand(me);

#pragma xmp loop on t0(i)
  for (int i = 0; i < n; i++){
    a0[i] = rand() % n;
  }

  xmp_sort_up(xmp_desc_of(a0), xmp_desc_of(b0));

#pragma xmp reflect (b0)

  result = 0;

#pragma xmp loop on t1(i) reduction(+:result)
  for (int i = 1; i < n; i++){
    if (b0[i-1] > b0[i]){
      result = 1;
    }
  }

  if (result > 0){
    if (me == 0) printf("ERROR\n");
    exit(1);
  }

  xmp_sort_down(xmp_desc_of(a0), xmp_desc_of(b0));

#pragma xmp reflect (b0)

  result = 0;

#pragma xmp loop on t1(i) reduction(+:result)
  for (int i = 1; i < n; i++){
    if (b0[i-1] < b0[i]){
      result = 1;
    }
  }

  if (result > 0){
    if (me == 0) printf("ERROR\n");
    exit(1);
  }

  //
  // float
  //

  //if (me == 0) printf("check for float starts...\n");

  float a1[n], b1[n];
#pragma xmp align a1[i] with t0(i)
#pragma xmp align b1[i] with t1(i)
#pragma xmp shadow b1[1:1]

  srand(me);

#pragma xmp loop on t0(i)
  for (int i = 0; i < n; i++){
    a1[i] = ((float)rand() / ((float)RAND_MAX + 1)) * 100;
  }

  xmp_sort_up(xmp_desc_of(a1), xmp_desc_of(b1));

#pragma xmp reflect (b1)

  result = 0;

#pragma xmp loop on t1(i) reduction(+:result)
  for (int i = 1; i < n; i++){
    if (b1[i-1] > b1[i]){
      result = 1;
    }
  }

  if (result > 0){
    if (me == 0) printf("ERROR\n");
    exit(1);
  }

  xmp_sort_down(xmp_desc_of(a1), xmp_desc_of(b1));

#pragma xmp reflect (b1)

  result = 0;

#pragma xmp loop on t1(i) reduction(+:result)
  for (int i = 1; i < n; i++){
    if (b1[i-1] < b1[i]){
      result = 1;
    }
  }

  if (result > 0){
    if (me == 0) printf("ERROR\n");
    exit(1);
  }

  //
  // double
  //

  //if (me == 0) printf("check for double starts...\n");

  double a2[n], b2[n];
#pragma xmp align a2[i] with t0(i)
#pragma xmp align b2[i] with t1(i)
#pragma xmp shadow b2[1:1]

  srand(me);

#pragma xmp loop on t0(i)
  for (int i = 0; i < n; i++){
    a2[i] = ((double)rand() / ((double)RAND_MAX + 1)) * 100;
  }

  xmp_sort_up(xmp_desc_of(a2), xmp_desc_of(b2));

#pragma xmp reflect (b2)

  result = 0;

#pragma xmp loop on t1(i) reduction(+:result)
  for (int i = 1; i < n; i++){
    if (b2[i-1] > b2[i]){
      result = 1;
    }
  }

  if (result > 0){
    if (me == 0) printf("ERROR\n");
    exit(1);
  }

  xmp_sort_down(xmp_desc_of(a2), xmp_desc_of(b2));

#pragma xmp reflect (b2)

  result = 0;

#pragma xmp loop on t1(i) reduction(+:result)
  for (int i = 1; i < n; i++){
    if (b2[i-1] < b2[i]){
      result = 1;
    }
  }

  if (result > 0){
    if (me == 0) printf("ERROR\n");
    exit(1);
  }

  if (me == 0) printf("PASS\n");
  return 0;

}
