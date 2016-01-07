#define TRUE 1
#define FALSE 0
#include <stdio.h>
#include <xmp.h>
#pragma xmp nodes p(2)
#define N 10
long a[N];
float b[N][N];
double c[N][N][N];
int d[N];
#pragma xmp coarray a,b,c,d : [*]
long a_test[N];
float b_test[N][N];
double c_test[N][N][N];
int d_test[N];
int status, return_val = 0;

void initialize_coarrays(int me)
{
  for(int i=0;i<N;i++){
    a[i] = i + me*100;
    a_test[i] = a[i];
  }

  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++){
      b[i][j] = i*N + j + me*100;
      b_test[i][j] = b[i][j];
    }

  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      for(int k=0;k<N;k++){
	c[i][j][k] = i*N*N + j*N + k + me*100;
	c_test[i][j][k] = c[i][j][k];
      }

  for(int i=0;i<N;i++){
    d[i] = i + me*100;
    d_test[i] = d[i];
  }

  xmp_sync_all(&status);
}

void test_1(int me)
{
  if(me == 2){
    long tmp;
    tmp = a[1]:[1];
    //    xmp_sync_memory(&status);
    a[1] = tmp;

    a_test[1] = 101;
  }

  xmp_sync_all(&status);
}

void check_1(int me){
  int i, flag = TRUE;
  
  for(i=0; i<N; i++){
    if( a[i] != a_test[i] ){
      flag = FALSE;
      printf("[%d] a[%d] check_1 : fall\ta[%d] = %ld (True value is %ld)\n",
	     me, i, i, a[i], a_test[i]);
    }
  }
  xmp_sync_all(&status);
  if(flag == TRUE)   printf("[%d] check_1 : PASS\n", me);
  else return_val = 1;
}

void test_2(int me)
{
  if(me == 1){
    float tmp[2][3];
    tmp[0:2][0] = b[1][1]:[2];

    b[0][0] = tmp[0][0];
    b[1][0] = tmp[1][0];
  }

  if(me == 1){
    b_test[0][0] = 211;
    b_test[1][0] = 211;
  }

  xmp_sync_all(&status);
}

void check_2(int me){
  int i, j, flag = TRUE;
  
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      if( b[i][j] != b_test[i][j] ){
	flag = FALSE;
	printf("[%d] b[%d][%d] check_2 : fall\tb[%d][%d] = %f (True value is %f)\n",
	       me, i, j, i, j, b[i][j], b_test[i][j]);
      }
    }
  }
  xmp_sync_all(&status);
  if(flag == TRUE)   printf("[%d] check_2 : PASS\n", me);
  else return_val = 1;
}

void test_3(int me){

  xmp_sync_all(&status);
  if(me == 2){
    c[0][1:3:3][2] = c[1][2][3]:[1];

  }
  
  if(me == 2){
    c_test[0][1][2] = 223;
    c_test[0][4][2] = 223;
    c_test[0][7][2] = 223;
  }

  xmp_sync_all(&status);
}

void check_3(int me){
  int i, j, k, flag = TRUE;
  
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      for(k=0; k<N; k++){
	if( c[i][j][k] != c_test[i][j][k] ){
	  flag = FALSE;
	  printf("[%d] c[%d][%d][%d] check_3 : fall\tc[%d][%d][%d] = %f (True value is %f)\n",
		 me, i, j, k, i, j, k, c[i][j][k], c_test[i][j][k]);
	}
      }
    }
  }
  xmp_sync_all(&status);
  if(flag == TRUE)   printf("[%d] check_3 : PASS\n", me);
  else return_val = 1;
}

void test_4(int me)
{
  if(me == 2){
    d[1:5] = d[4]:[1];

    d_test[1] = 104;
    d_test[2] = 104;
    d_test[3] = 104;
    d_test[4] = 104;
    d_test[5] = 104;
  }

  xmp_sync_all(&status);
}

void check_4(int me){
  int i, flag = TRUE;
  
  for(i=0; i<N; i++){
    if( d[i] != d_test[i] ){
      flag = FALSE;
      printf("[%d] d[%d] check_4 : fall\td[%d] = %d (True value is %d)\n",
	     me, i, i, d[i], d_test[i]);
    }
  }
  xmp_sync_all(&status);
  if(flag == TRUE)   printf("[%d] check_4 : PASS\n", me);
  else return_val = 1;
}

int main(){
  int me = xmp_node_num();
  
  initialize_coarrays(me);
  
  test_1(me);
  check_1(me);
  test_2(me);
  check_2(me);
  test_3(me);
  check_3(me);
  test_4(me);
  check_4(me);

#pragma xmp barrier
#pragma xmp reduction(MAX:return_val)
  return return_val;
}
