#include "xmp.h"
#include <stdlib.h>
#include <stdio.h>

int me;
int nprocs;

int n = 1024;
int *m;

#pragma xmp nodes p(*)

#pragma xmp template t0(0:n-1)
#pragma xmp distribute t0(block) onto p

int a0[n];
#pragma xmp align a0[i] with t0(i)

float b0[n];
#pragma xmp align b0[i] with t0(i)

double c0[n];
#pragma xmp align c0[i] with t0(i)

void int_block();
void float_gblock();
void double_cyclic();

int main(int argc, char *argv[]){

  me = xmp_node_num() - 1;
  nprocs = xmp_num_nodes();

  m = (int *)malloc(nprocs * sizeof(int));

  int p = 0;

  for (int i = 0; i < nprocs - 1; i++){
    m[i] = nprocs * 2 * (i + 1);
    p = p + m[i];
  }

  m[nprocs - 1] = n - p;

  int_block();
  float_gblock();
  double_cyclic();

  if (me == 0) printf("PASS\n");
  return 0;

}


void int_block(){

  int result;

#pragma xmp template t1(0:n-1)
#pragma xmp distribute t1(block) onto p

  int a1[n];
#pragma xmp align a1[i] with t1(i)
#pragma xmp shadow a1[1:1]

  srand(me);

#pragma xmp loop on t0(i)
  for (int i = 0; i < n; i++){
    a0[i] = rand() % n;
  }

  xmp_sort_up(xmp_desc_of(a0), xmp_desc_of(a1));

#pragma xmp reflect (a1)

  result = 0;

#pragma xmp loop on t1(i) reduction(+:result)
  for (int i = 1; i < n; i++){
    if (a1[i-1] > a1[i]){
      result = 1;
      break;
    }
  }

  if (result > 0){
    if (me == 0) printf("ERROR\n");
    exit(1);
  }

  xmp_sort_down(xmp_desc_of(a0), xmp_desc_of(a1));

#pragma xmp reflect (a1)

  result = 0;

#pragma xmp loop on t1(i) reduction(+:result)
  for (int i = 1; i < n; i++){
    if (a1[i-1] < a1[i]){
      result = 1;
      break;
    }
  }

  if (result > 0){
    if (me == 0) printf("ERROR\n");
    exit(1);
  }

}


void float_gblock(){

  int result;

#pragma xmp template t1(0:n-1)
#pragma xmp distribute t1(gblock(m)) onto p

  float b1[n];
#pragma xmp align b1[i] with t1(i)
#pragma xmp shadow b1[1:1]

  srand(me);

#pragma xmp loop on t0(i)
  for (int i = 0; i < n; i++){
    b0[i] = ((float)rand() / ((float)RAND_MAX + 1)) * 100;
  }

  xmp_sort_up(xmp_desc_of(b0), xmp_desc_of(b1));

#pragma xmp reflect (b1)

  result = 0;

#pragma xmp loop on t1(i) reduction(+:result)
  for (int i = 1; i < n; i++){
    if (b1[i-1] > b1[i]){
      result = 1;
      break;
    }
  }

  if (result > 0){
    if (me == 0) printf("ERROR\n");
    exit(1);
  }

  xmp_sort_down(xmp_desc_of(b0), xmp_desc_of(b1));

#pragma xmp reflect (b1)

  result = 0;

#pragma xmp loop on t1(i) reduction(+:result)
  for (int i = 1; i < n; i++){
    if (b1[i-1] < b1[i]){
      result = 1;
      break;
    }
  }

  if (result > 0){
    if (me == 0) printf("ERROR\n");
    exit(1);
  }

}


void double_cyclic(){

  int result;

#pragma xmp template t1(0:n-1)
#pragma xmp distribute t1(cyclic(4)) onto p

  double c1[n];
#pragma xmp align c1[i] with t1(i)

  double d1[n];
#pragma xmp align d1[i] with t0(i)
#pragma xmp shadow d1[1:1]


  srand(me);

#pragma xmp loop on t0(i)
  for (int i = 0; i < n; i++){
    c0[i] = ((double)rand() / ((double)RAND_MAX + 1)) * 100;
  }

  xmp_sort_up(xmp_desc_of(c0), xmp_desc_of(c1));

#pragma xmp gmove
  d1[:] = c1[:];

#pragma xmp reflect (d1)

  result = 0;

#pragma xmp loop on t0(i) reduction(+:result)
  for (int i = 1; i < n; i++){
    if (d1[i-1] > d1[i]){
      result = 1;
      break;
    }
  }

  if (result > 0){
    if (me == 0) printf("ERROR\n");
    exit(1);
  }

  xmp_sort_down(xmp_desc_of(c0), xmp_desc_of(c1));

#pragma xmp gmove
  d1[:] = c1[:];

#pragma xmp reflect (d1)

  result = 0;

#pragma xmp loop on t1(i) reduction(+:result)
  for (int i = 1; i < n; i++){
    if (d1[i-1] < d1[i]){
      result = 1;
      break;
    }
  }

  if (result > 0){
    if (me == 0) printf("ERROR\n");
    exit(1);
  }

}
