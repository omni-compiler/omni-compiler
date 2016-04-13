#include <stdio.h>
#include <stdlib.h>
#include "xmp.h"

#define N 8

#pragma xmp nodes p(2)

#pragma xmp template t1(0:N-1)
#pragma xmp distribute t1(block) onto p

int a[N];
#pragma xmp align a[i] with t1(i)
#pragma xmp shadow a[2:1]

#pragma xmp template t2(0:N-1)
#pragma xmp distribute t2(cyclic) onto p

int b[N];
#pragma xmp align b[i] with t2(i)


//--------------------------------------------------------
void gmove_in(){

  int result = 0;

#pragma xmp loop (i) on t1(i)
  for (int i = 0; i < N; i++){
    a[i] = 777;
  }

#pragma xmp loop (i) on t2(i)
  for (int i = 0; i < N; i++){
    b[i] = i;
  }

#pragma xmp barrier

#ifdef _MPI3
#pragma xmp gmove in
  a[:] = b[:];
#endif

#pragma xmp loop (i) on t1(i) reduction(+:result)
  for (int i = 0; i < N; i++){
    if (a[i] != i){
      result = 1;
      printf("(%d), %d, %d\n", xmp_node_num(), i, a[i]);
    }
  }

#pragma xmp task on p(1)
  {
    if (result != 0){
      printf("ERROR in gmove_in\n");
      exit(1);
    }
  }

}


//--------------------------------------------------------
void gmove_in_async(){

  int result = 0;

#pragma xmp loop (i) on t1(i)
  for (int i = 0; i < N; i++){
     a[i] = 777;
  }

#pragma xmp loop (i) on t2(i)
  for (int i = 0; i < N; i++){
    b[i] = i;
  }

#pragma xmp barrier

#ifdef _MPI3
#pragma xmp gmove in async(10)
  a[:] = b[:];
#endif

#pragma xmp wait_async(10)

#pragma xmp loop (i) on t1(i) reduction(+:result)
  for (int i = 0; i < N; i++){
    if (a[i] != i){
      result = 1;
      //printf("(%d), %d, %d\n", xmp_node_num(), i, a[i]);
    }
  }

#pragma xmp task on p(1)
  {
    if (result != 0){
      printf("ERROR in gmove_in_async\n");
      exit(1);
    }
  }

}


//--------------------------------------------------------
void gmove_out(){

  int result = 0;

#pragma xmp loop (i) on t1(i)
  for (int i = 0; i < N; i++){
    a[i] = 777;
  }

#pragma xmp loop (i) on t2(i)
  for (int i = 0; i < N; i++){
    b[i] = i;
  }

#pragma xmp barrier

#ifdef _MPI3
#pragma xmp gmove out
  a[:] = b[:];
#endif

#pragma xmp loop (i) on t1(i) reduction(+:result)
  for (int i = 0; i < N; i++){
    if (a[i] != i){
      result = 1;
      //printf("(%d), %d, %d\n", xmp_node_num(), i, a[i]);
    }
  }

#pragma xmp task on p(1)
  {
    if (result != 0){
      printf("ERROR in gmove_out\n");
      exit(1);
    }
  }

}


//--------------------------------------------------------
void gmove_out_async(){

  int result = 0;

#pragma xmp loop (i) on t1(i)
  for (int i = 0; i < N; i++){
    a[i] = 777;
  }

#pragma xmp loop (i) on t2(i)
  for (int i = 0; i < N; i++){
    b[i] = i;
  }

#pragma xmp barrier

#ifdef _MPI3
#pragma xmp gmove out async(10)
  a[:] = b[:];
#endif

#pragma xmp wait_async(10)

#pragma xmp loop (i) on t1(i) reduction(+:result)
  for (int i = 0; i < N; i++){
    if (a[i] != i){
      result = 1;
      //printf("(%d), %d, %d\n", xmp_node_num(), i, a[i]);
    }
  }

#pragma xmp task on p(1)
  {
    if (result != 0){
      printf("ERROR in gmove_out_async\n");
      exit(1);
    }
  }

}


//--------------------------------------------------------
int main(){

#ifdef _MPI3
  gmove_in();
  gmove_in_async();
  gmove_out();
  gmove_out_async();

#pragma xmp task on p(1)
  {
    printf("PASS\n");
  }
#else
  printf("Skipped\n");
#endif

  return 0;

}
