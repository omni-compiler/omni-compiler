#define TRUE 1
#define FALSE 0
#include <stdio.h>
#include <string.h>
#include "xmp.h"
int a[10]:[*], a_test[10];
float b[3][5]:[*], b_test[3][5];
double c[2][3][4]:[*], c_test[2][3][4];
long d[3][4][3][2]:[*], d_test[3][4][3][2];
int e[5][7]:[*], e_test[5][7];
int return_val = 0;
#pragma xmp nodes p[2]

void initialize(int me){
  int i, j, m, n, t = me * 100;
  
  for(i=0;i<10;i++){
    a[i] = i + t;
    a_test[i] = a[i];
  }
  
  for(i=0;i<3;i++){
    for(j=0;j<5;j++){
      b[i][j] = 5*i + j + t;
      b_test[i][j] = b[i][j];
    }
  }
	
  for(i=0;i<2;i++){
    for(j=0;j<3;j++){
      for(m=0;m<4;m++){
	c[i][j][m] = 12*i + 4*j + m + t;
	c_test[i][j][m] = c[i][j][m];
      }
    }
  }
  
  for(i=0;i<3;i++){
    for(j=0;j<4;j++){
      for(m=0;m<3;m++){
	for(n=0;n<2;n++){
	  d[i][j][m][n] = 12*i + 6 * j + 2 * m + n + t;
	  d_test[i][j][m][n] = d[i][j][m][n];
	}
      }
    }
  }

  for(i=0;i<5;i++){
    for(j=0;j<7;j++){
      e[i][j] = 9*i + j + t;
      e_test[i][j] = e[i][j];
    }
  }
}

void communicate_1(int me){
  xmp_sync_all(NULL);
  if(me == 1){
    int tmp[100];
    tmp[3:5] = a[2:5]:[0]; // get
    memcpy(&a[3], &tmp[3], sizeof(int)*5);
  }
  if(me == 0){
    int tmp[50];
    tmp[8] = 999; tmp[9] = 1000;
    a[0:2]:[1] = tmp[8:2]; // put
  }
  
  if(me == 1){
    a_test[3] = 2; a_test[4] = 3; a_test[5] = 4;
    a_test[6] = 5; a_test[7] = 6; 
    a_test[0] = 999; a_test[1] = 1000;
  }
  
  xmp_sync_all(NULL);
}

void check_1(int me){
  int i, flag = TRUE;
  
  for(i=0; i<10; i++){
    if( a[i] != a_test[i] ){
      flag = FALSE;
      printf("[%d] a[%d] check_1 : fall\ta[%d] = %d (True value is %d)\n",
	     me, i, i, a[i], a_test[i]);
    }
  }
  xmp_sync_all(NULL);
  if(flag == TRUE)   printf("[%d] check_1 : PASS\n", me);
  else return_val = 1;
}

void communicate_2(int me){
  xmp_sync_all(NULL);
  if(me == 0){
    int a1, a2, a3, a4, a5, a6, a7;
    a1 = 1; a2 = 2; a3 = 3;
    a4 = 0; a5 = 1; a6 = 3;
    a7 = 1;
    b[a1][a2:a3] = b[a4][a5:a6]:[a7];  // get
    b[2][:1]:[1] = b[0][:1];           // put
  }
  
  if(me == 0){
    b_test[1][2] = 101; b_test[1][3] = 102; b_test[1][4] = 103;
  }
  if(me == 1){
    b_test[2][0] = 0;
  }
  
  xmp_sync_all(NULL);
}

void check_2(int me){
  xmp_sync_all(NULL);
  int i, j, flag = TRUE;
  
  for(i=0;i<3;i++){
    for(j=0;j<5;j++){
      if( b[i][j] != b_test[i][j] ){
	flag = FALSE;
	printf("[%d] b[%d][%d] check_2 : fall\tb[%d][%d] = %.f (True value is %.f)\n",
	       me, i, j, i, j, b[i][j], b_test[i][j]);
      }
    }
  }
  xmp_sync_all(NULL);
  if(flag == TRUE)   printf("[%d] check_2 : PASS\n", me);
  else return_val = 1;
}

void communicate_3(int me){
  xmp_sync_all(NULL);
  if(me == 1){
    c[1][2][0:1]:[0] = c[1][1][1];     // put
  }
  if(me == 0){
    double tmp[2][5];
    tmp[1][0:5] = c[0][2][1:5]:[1];       // get
    memcpy(&c[0][2][1], &tmp[1][0], sizeof(double) * 3);
  }
  
  if(me == 0){
    c_test[1][2][0] = 117;
    c_test[0][2][1] = 109; c_test[0][2][2] = 110; c_test[0][2][3] = 111;
  }
  
  xmp_sync_all(NULL);
}

void check_3(int me){
  xmp_sync_all(NULL);
  int i, j, m, flag = TRUE;

  for(i=0;i<2;i++){
    for(j=0;j<3;j++){
      for(m=0;m<4;m++){
	if( c[i][j][m] != c_test[i][j][m] ){
	  flag = FALSE;
	  printf("[%d] c[%d][%d][%d] check_3 : fall\tc[%d][%d][%d] = %.f (True value is %.f)\n",
		 me, i, j, m, i, j, m, c[i][j][m], c_test[i][j][m]);
	}
      }
    }
  }
  xmp_sync_all(NULL);
  if(flag == TRUE)   printf("[%d] check_3 : PASS\n", me);
  else return_val = 1;
}

void communicate_4(int me){
  xmp_sync_all(NULL);
  if(me == 1){
    long tmp[2] = {5, 9};
    d[1][2][1][:]:[0] = tmp[:];          // put
  }
  if(me == 1){
    d[0][1][1][:] = d[0][2][1][:]:[0];   // get 
  }

  if(me == 0){
    d_test[1][2][1][0] = 5; d_test[1][2][1][1] = 9;
  }
  if(me == 1){
    d_test[0][1][1][0] = 14; d_test[0][1][1][1] = 15;
  }

  xmp_sync_all(NULL);
}


void check_4(int me){
  xmp_sync_all(NULL);

  int i, j, m, n, flag = TRUE;
  for(i=0;i<3;i++){
    for(j=0;j<4;j++){
      for(m=0;m<3;m++){
	for(n=0;n<2;n++){
	  if( d[i][j][m][n] != d_test[i][j][m][n] ){
	    flag = FALSE;
	    printf("[%d] d[%d][%d][%d][%d] check_3 : fall\td[%d][%d][%d][%d] = %ld (True value is %ld)\n",
		   me, i, j, m, n, i, j, m, n, d[i][j][m][n], d_test[i][j][m][n]);
	  }
        }
      }
    }
  }

  xmp_sync_all(NULL);
  if(flag == TRUE)   printf("[%d] check_4 : PASS\n", me);
  else return_val = 1;
}

void communicate_5(int me){
  xmp_sync_all(NULL);
  const int e1 = 1, e2 = 2;
  const int e3 = 2, e4 = 3;

  if(me == 0){
    e[e1+1][e2+1:2]:[1] = e[e3+1][e4+1:2];
    e_test[3][5] = 132;
    e_test[4][5] = 141;
  }

  if(me == 1){
    e[3:2][e4+2]:[0] = e[3:2][e4+2];
    e_test[2][3] = 31;
    e_test[2][4] = 32;
  }
  xmp_sync_all(NULL);
}

void check_5(int me){
  xmp_sync_all(NULL);
  int i, j, flag = TRUE;

  for(i=0;i<5;i++){
    for(j=0;j<7;j++){
      if( e[i][j] != e_test[i][j] ){
	flag = FALSE;
	printf("[%d] b[%d][%d] check_5 : fall\tb[%d][%d] = %d (True value is %d)\n",
	       me, i, j, i, j, e[i][j], e_test[i][j]);
      }
    }
  }
  xmp_sync_all(NULL);
  if(flag == TRUE)   printf("[%d] check_5 : PASS\n", me);
  else return_val = 1;
}

void bug_107()
{
  
}

int main(){
  int me = xmpc_this_image();
  initialize(me);
  
  communicate_1(me);
  check_1(me);
  
  communicate_2(me);
  check_2(me);
  
  communicate_3(me);
  check_3(me);
  
  communicate_4(me);
  check_4(me);

  communicate_5(me);
  check_5(me);

  bug_107();
  
#pragma xmp barrier
#pragma xmp reduction(MAX:return_val)
  return return_val;
}
