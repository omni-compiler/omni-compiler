#include <stdio.h>
#include <stdlib.h>

#define N 64

#pragma xmp nodes p0(8)

#pragma xmp nodes p1(2,2) = p0(1:4)
#pragma xmp nodes p2(2,2) = p0(5:8)

#pragma xmp template t1(0:N-1,0:N-1,0:N-1)
#pragma xmp distribute t1(*,block,block) onto p1

int a[N][N][N];
#pragma xmp align a[i][j][k] with t1(i,j,k)
#pragma xmp shadow a[0][2:1][1:0]

#pragma xmp template t2(0:N-1,0:N-1,0:N-1)
#pragma xmp distribute t2(block,cyclic,*) onto p2

int b[N][N][N];
#pragma xmp align b[i][j][k] with t2(i,j,k)
#pragma xmp shadow b[0:1][0][0]

int x[N][N][N];

int s;

void init_a(){
#pragma xmp loop (i,j,k) on t1(i,j,k)
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      for (int k = 0; k < N; k++){
	a[i][j][k] = 777;
      }
    }
  }
}

void init_b(){
#pragma xmp loop (i,j,k) on t2(i,j,k)
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      for (int k = 0; k < N; k++){
	b[i][j][k] = i*10000 + j *100 + k;
      }
    }
  }
}

void init_x(){
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      for (int k = 0; k < N; k++){
	x[i][j][k] = i*10000 + j *100 + k;
      }
    }
  }
}

void init_x0(){
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      for (int k = 0; k < N; k++){
	x[i][j][k] = 0;
      }
    }
  }
}


//--------------------------------------------------------
// global section = global section
//--------------------------------------------------------
void gmove_gs_gs(){

  int result = 0;

  init_a();
  init_b();

#pragma xmp barrier

#pragma xmp task on p2
  {

#ifdef _MPI3
#pragma xmp gmove out
  a[0:N/4][N/2:N/2][4:N-5] = b[N/2:N/4:2][0:N/2][0:N-5];
#endif

  }

#pragma xmp barrier

#pragma xmp task on p1
  {

#pragma xmp loop (i,j,k) on t1(i,j,k) reduction (+:result)
  for (int i = 0; i < N/4; i++){
    for (int j = N/2; j < N; j++){
      for (int k = 4; k < N-1; k++){
	if (a[i][j][k] != (N/2+i*2)*10000 + (j-N/2)*100 + (k-4)){
	  result = 1;
	}
      }
    }
  }

#pragma xmp task on p1(1,1) nocomm
  {
    if (result != 0){
      printf("ERROR in gmove_gs_gs\n");
      exit(1);
    }
  }

  }

}


//--------------------------------------------------------
// global section = global element
//--------------------------------------------------------
void gmove_gs_ge(){

  int result = 0;

  init_a();
  init_b();

#pragma xmp barrier

#pragma xmp task on p2
  {

#ifdef _MPI3
#pragma xmp gmove out
  a[0:N/4][N/2:N/2][4:N-5] = b[3][4][5];
#endif

  }

#pragma xmp barrier

#pragma xmp task on p1
  {

#pragma xmp loop (i,j,k) on t1(i,j,k) reduction(+:result)
  for (int i = 0; i < N/4; i++){
    for (int j = N/2; j < N; j++){
      for (int k = 4; k < N-1; k++){
	if (a[i][j][k] != 3*10000 + 4*100 + 5){
	  result = 1;
	}
      }
    }
  }

#pragma xmp task on p1(1,1) nocomm
  {
    if (result != 0){
      printf("ERROR in gmove_gs_ge\n");
      exit(1);
    }
  }

  }

}

//--------------------------------------------------------
// global element = global element
//--------------------------------------------------------
void gmove_ge_ge(){

  init_a();
  init_b();

#pragma xmp barrier

#pragma xmp task on p2
  {

#ifdef _MPI3
#pragma xmp gmove out
  a[7][8][9] = b[3][4][5];
#endif

  }

#pragma xmp barrier

#pragma xmp task on p1
  {

#pragma xmp task on t1(7,8,9) nocomm
  {
    if (a[7][8][9] != 3*10000 + 4*100 + 5){
      printf("ERROR in gmove_ge_ge\n");
      exit(1);
    }
  }

  }

}


//--------------------------------------------------------
// global section = local section
//--------------------------------------------------------
void gmove_gs_ls(){

  int result = 0;

  init_a();
  init_x();

#pragma xmp barrier

#pragma xmp task on p2
  {

#ifdef _MPI3
#pragma xmp gmove out
  a[0:N/4][N/2:N/2][4:N-5] = b[N/2:N/4:2][0:N/2][0:N-5];
#endif

  }

#pragma xmp barrier

#pragma xmp task on p1
  {

#pragma xmp loop (i,j,k) on t1(i,j,k) reduction(+:result)
  for (int i = 0; i < N/4; i++){
    for (int j = N/2; j < N; j++){
      for (int k = 4; k < N-1; k++){
	if (a[i][j][k] != (N/2+i*2)*10000 + (j-N/2)*100 + (k-4)){
	  result = 1;
	}
      }
    }
  }

#pragma xmp task on p1(1,1) nocomm
  {
    if (result != 0){
      printf("ERROR in gmove_gs_ls\n");
      exit(1);
    }
  }

  }

}


//--------------------------------------------------------
// global section = local element
//--------------------------------------------------------
void gmove_gs_le(){

  int result = 0;

  init_a();
  init_x();

#pragma xmp barrier

#pragma xmp task on p2
  {

#ifdef _MPI3
#pragma xmp gmove out
  a[0:N/4][N/2:N/2][4:N-5] = x[3][4][5];
#endif

  }

#pragma xmp barrier

#pragma xmp task on p1
  {

#pragma xmp loop (i,j,k) on t1(i,j,k) reduction(+:result)
  for (int i = 0; i < N/4; i++){
    for (int j = N/2; j < N; j++){
      for (int k = 4; k < N-1; k++){
	if (a[i][j][k] != 3*10000 + 4*100 + 5){
	  result = 1;
	}
      }
    }
  }

#pragma xmp task on p1(1,1) nocomm
  {
    if (result != 0){
      printf("ERROR in gmove_gs_le\n");
      exit(1);
    }
  }

  }

}

//--------------------------------------------------------
// global element = local element
//--------------------------------------------------------
void gmove_ge_le(){

  init_a();
  init_x();

#pragma xmp barrier

#pragma xmp task on p2
  {

#ifdef _MPI3
#pragma xmp gmove out
  a[7][8][9] = x[3][4][5];
#endif

  }

#pragma xmp barrier

#pragma xmp task on p1
  {

#pragma xmp task on t1(7,8,9) nocomm
  {
    if (a[7][8][9] != 3*10000 + 4*100 + 5){
      printf("ERROR in gmove_ge_le\n");
      exit(1);
    }
  }

  }

}


//--------------------------------------------------------
// global section = scalar
//--------------------------------------------------------
void gmove_gs_s(){

  int result = 0;

  init_a();
  s = 111;

#pragma xmp barrier

#pragma xmp task on p2
  {

#ifdef _MPI3
#pragma xmp gmove out
  a[0:N/4][N/2:N/2][4:N-5] = s;
#endif

  }

#pragma xmp barrier

#pragma xmp task on p1
  {

#pragma xmp loop (i,j,k) on t1(i,j,k) reduction(+:result)
  for (int i = 0; i < N/4; i++){
    for (int j = N/2; j < N; j++){
      for (int k = 4; k < N-1; k++){
	if (a[i][j][k] != s){
	  result = 1;
	}
      }
    }
  }

#pragma xmp task on p1(1,1) nocomm
  {
    if (result != 0){
      printf("ERROR in gmove_gs_s\n");
      exit(1);
    }
  }

  }

}

//--------------------------------------------------------
// global element = scalar
//--------------------------------------------------------
void gmove_ge_s(){

  init_a();
  s = 111;

#pragma xmp barrier

#pragma xmp task on p2
  {

#ifdef _MPI3
#pragma xmp gmove out
  a[7][8][9] = s;
#endif

  }

#pragma xmp barrier

#pragma xmp task on p1
  {

#pragma xmp task on t1(7,8,9) nocomm
  {
    if (a[7][8][9] != s){
      printf("ERROR in gmove_ge_s\n");
      exit(1);
    }
  }

  }

}


//--------------------------------------------------------
int main(){

#ifdef _MPI3
  gmove_gs_gs();
  gmove_gs_ge();
  gmove_ge_ge();

  gmove_gs_ls();
  gmove_gs_le();
  gmove_ge_le();

  gmove_gs_s();
  gmove_ge_s();

#pragma xmp task on p0(1) nocomm
  {
    printf("PASS\n");
  }
#else
#pragma xmp task on p0(1) nocomm
  {
    printf("Skipped\n");
  }
#endif

}
