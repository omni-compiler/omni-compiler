#define TRUE 1
#define FALSE 0
#include <stdio.h>
#include <string.h>
#include "xmp.h"
int a[20]:[*], a_test[20];
float b[4][10]:[*], b_test[4][10];
double c[2][10][10]:[*], c_test[2][10][10];
long d[3][2][3][2]:[*], d_test[3][2][3][2];
int status, return_val = 0;
#pragma xmp nodes p(2)


void initialize(int me){
  int i, j, m, n, t = (me-1) * 100;
  
  for(i=0;i<20;i++){
    a[i] = i + t;
    a_test[i] = a[i];
  }
  
  for(i=0;i<4;i++){
    for(j=0;j<10;j++){
      b[i][j] = 10*i + j + t;
      b_test[i][j] = b[i][j];
    }
  }
	
  for(i=0;i<2;i++){
    for(j=0;j<10;j++){
      for(m=0;m<10;m++){
	c[i][j][m] = 100*i + 10*j + m + t;
	c_test[i][j][m] = c[i][j][m];
      }
    }
  }
  
  for(i=0;i<3;i++){
    for(j=0;j<2;j++){
      for(m=0;m<3;m++){
	for(n=0;n<2;n++){
	  d[i][j][m][n] = 12*i + 6 * j + 2 * m + n + t;
	  d_test[i][j][m][n] = d[i][j][m][n];
	}
      }
    }
  }
}

void communicate_1(int me){
  xmp_sync_all(&status);
  if(me == 2){
    int tmp[100];
    tmp[3:5:2] = a[2:5]:[1]; // get
    a[3] = tmp[3]; a[5] = tmp[5]; a[7] = tmp[7];
    a[9] = tmp[9]; a[11] = tmp[11];
  }
  if(me == 1){
    int tmp[50];
    tmp[8] = 999; tmp[10] = 1000;
    a[0:2]:[2] = tmp[8:2:2]; // put
  }
  
  if(me == 2){
    a_test[3] = 2; a_test[5] = 3; a_test[7] = 4;
    a_test[9] = 5; a_test[11] = 6; 
    a_test[0] = 999; a_test[1] = 1000;
  }
  
  xmp_sync_all(&status);
}

void check_1(int me){
  int i, flag = TRUE;
  
  for(i=0; i<20; i++){
    if( a[i] != a_test[i] ){
      flag = FALSE;
      printf("[%d] a[%d] check_1 : fall\ta[%d] = %d (True value is %d)\n",
	     me, i, i, a[i], a_test[i]);
    }
  }
  xmp_sync_all(&status);
  if(flag == TRUE)   printf("[%d] check_1 : PASS\n", me);
  else return_val = 1;
}

void communicate_2(int me){
  xmp_sync_all(&status);
  if(me == 1){
    b[2][2:3:2]:[2] = b[1][1:3];  // put
    b[3][2:3] = b[2][1:3:2]:[2];  // get
  }
  if(me == 1){
    b_test[3][2] = 121; b_test[3][3] = 123; b_test[3][4] = 125;
  }
  if(me == 2){
    b_test[2][2] = 11;
    b_test[2][4] = 12;
    b_test[2][6] = 13;
  }
  
  xmp_sync_all(&status);
}

void check_2(int me){
  xmp_sync_all(&status);
  int i, j, flag = TRUE;
  
  for(i=0;i<4;i++){
    for(j=0;j<10;j++){
      if( b[i][j] != b_test[i][j] ){
	flag = FALSE;
	printf("[%d] b[%d][%d] check_2 : fall\tb[%d][%d] = %.f (True value is %.f)\n",
	       me, i, j, i, j, b[i][j], b_test[i][j]);
      }
    }
  }
  xmp_sync_all(&status);
  if(flag == TRUE)   printf("[%d] check_2 : PASS\n", me);
  else return_val = 1;
}

void communicate_3(int me){
  xmp_sync_all(&status);
  if(me == 2){
    c[1][2:2:3][0:1:2]:[1] = c[1][1:2:5][1];   // put
  }
  if(me == 1){
    double tmp[5][5];
    tmp[0:3:2][0] = c[0][1:3:2][2]:[2];       // get
    
    c[0][1][0] = tmp[0][0];
    c[0][3][0] = tmp[2][0];
    c[0][5][0] = tmp[4][0];
  }
  
  if(me == 1){
    c_test[1][2][0] = 111 + 100;
    c_test[1][5][0] = 161 + 100;
    c_test[0][1][0] = 112; c_test[0][3][0] = 132; c_test[0][5][0] = 152;
  }
  
  xmp_sync_all(&status);
}

void check_3(int me){
  xmp_sync_all(&status);
  int i, j, m, flag = TRUE;

  for(i=0;i<2;i++){
    for(j=0;j<10;j++){
      for(m=0;m<10;m++){
	if( c[i][j][m] != c_test[i][j][m] ){
	  flag = FALSE;
	  printf("[%d] c[%d][%d][%d] check_3 : fall\tc[%d][%d][%d] = %.f (True value is %.f)\n",
		 me, i, j, m, i, j, m, c[i][j][m], c_test[i][j][m]);
	}
      }
    }
  }
  xmp_sync_all(&status);
  if(flag == TRUE)   printf("[%d] check_3 : PASS\n", me);
  else return_val = 1;
}

int main(){
  int me;
  
  me = xmp_node_num();
  initialize(me);
  
  communicate_1(me);
  check_1(me);

  communicate_2(me);
  check_2(me);
  
  communicate_3(me);
  check_3(me);

#pragma xmp barrier
#pragma xmp reduction(MAX:return_val)  
  return return_val;
}
