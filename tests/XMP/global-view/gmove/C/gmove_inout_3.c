#include <stdio.h>
#include <stdlib.h>
#include "xmp.h"

#define N 4

#pragma xmp nodes p0[8]

#pragma xmp nodes p1[2][2] = p0[0:4]
#pragma xmp nodes p2[2][2] = p0[4:4]

#pragma xmp template t1[N][N][N]
#pragma xmp distribute t1[block][block][*] onto p1

int a[N][N][N];
#pragma xmp align a[i][j][k] with t1[k][j][i]
#pragma xmp shadow a[0][2:1][1:0]

#pragma xmp template t2[N][N][N]
#pragma xmp distribute t2[*][cyclic][block] onto p2

int b[N][N][N];
#pragma xmp align b[i][j][k] with t2[k][j][i]
#pragma shadow b[0:1][0][0]


//--------------------------------------------------------
void gmove_in(){

  int result = 0;

#pragma xmp loop on t1[k][j][i]
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      for (int k = 0; k < N; k++){
	a[i][j][k] = 777;
      }
    }
  }

#pragma xmp loop on t2[k][j][i]
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      for (int k = 0; k < N; k++){
	b[i][j][k] = i*10000 + j *100 + k;
      }
    }
  }

#pragma xmp barrier

#pragma xmp task on p1
  {
#ifdef _MPI3
#pragma xmp gmove in
    a[:][:][:] = b[:][:][:];
#endif
  }

#pragma xmp barrier

#pragma xmp loop on t1[k][j][i] reduction(+:result)
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      for (int k = 0; k < N; k++){
	if (a[i][j][k] != i*10000 + j*100 + k){
	  //printf("(%d) %d %d %d %d\n", xmp_node_num(), i, j, k, a[i][j][k]);
	  result = 1;
	}
      }
    }
  }

#pragma xmp task on p0[0]
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

#pragma xmp loop on t1[k][j][i]
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      for (int k = 0; k < N; k++){
	a[i][j][k] = 777;
      }
    }
  }

#pragma xmp loop on t2[k][j][i]
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      for (int k = 0; k < N; k++){
	b[i][j][k] = i*10000 + j *100 + k;
      }
    }
  }

#pragma xmp barrier

#pragma xmp task on p1
  {
#ifdef _MPI3
#pragma xmp gmove in async(10)
    a[:][:][:] = b[:][:][:];
#endif

#pragma xmp wait_async (10)
  }

#pragma xmp barrier

#pragma xmp loop on t1[k][j][i] reduction(+:result)
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      for (int k = 0; k < N; k++){
	if (a[i][j][k] != i*10000 + j*100 + k){
	  //printf("(%d) %d %d %d %d\n", xmp_node_num(), i, j, k, a[i][j][k]);
	  result = 1;
	}
      }
    }
  }

#pragma xmp task on p0[0]
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

#pragma xmp loop on t1[k][j][i]
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      for (int k = 0; k < N; k++){
	a[i][j][k] = 777;
      }
    }
  }

#pragma xmp loop on t2[k][j][i]
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      for (int k = 0; k < N; k++){
	b[i][j][k] = i*10000 + j *100 + k;
      }
    }
  }

#pragma xmp barrier

#pragma xmp task on p2
  {
#ifdef _MPI3
#pragma xmp gmove out
    a[:][:][:] = b[:][:][:];
#endif
  }

#pragma xmp barrier

#pragma xmp loop on t1[k][j][i] reduction(+:result)
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      for (int k = 0; k < N; k++){
	if (a[i][j][k] != i*10000 + j*100 + k){
	  //printf("(%d) %d %d %d %d\n", xmp_node_num(), i, j, k, a[i][j][k]);
	  result = 1;
	}
      }
    }
  }

#pragma xmp task on p0[0]
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

#pragma xmp loop on t1[k][j][i]
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      for (int k = 0; k < N; k++){
	a[i][j][k] = 777;
      }
    }
  }

#pragma xmp loop on t2[k][j][i]
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      for (int k = 0; k < N; k++){
	b[i][j][k] = i*10000 + j *100 + k;
      }
    }
  }

#pragma xmp barrier

#pragma xmp task on p2
  {
#ifdef _MPI3
#pragma xmp gmove out async(10)
    a[:][:][:] = b[:][:][:];
#endif

#pragma xmp wait_async (10)
  }

#pragma xmp barrier

#pragma xmp loop on t1[k][j][i] reduction(+:result)
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      for (int k = 0; k < N; k++){
	if (a[i][j][k] != i*10000 + j*100 + k){
	  //printf("(%d) %d %d %d %d\n", xmp_node_num(), i, j, k, a[i][j][k]);
	  result = 1;
	}
      }
    }
  }

#pragma xmp task on p0[0]
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

#pragma xmp task on p0[0]
  {
    printf("PASS\n");
  }
#else
  printf("Skipped\n");
#endif

  return 0;

}
