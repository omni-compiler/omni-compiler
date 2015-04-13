#define TRUE 1
#define FALSE 0
#include <stdio.h>
#include <xmp.h>
#pragma xmp nodes p(2)
int a:[*], a_test;
long b[2]:[*], b_test[2];
float c[2][3]:[*], c_test[2][3];
double d[2][3][4]:[*], d_test[2][3][4];

int status, return_val = 0;

void initialize_coarrays(int me){
  int i, j, k, n = 100 * (me - 1);

  a = n + 1;
  a_test = a;
  
  for(i=0;i<2;i++){
    b[i] = i + n;
    b_test[i] = b[i];
  }
  
  for(i=0;i<2;i++){
    for(j=0;j<3;j++){
      c[i][j] = i*3 + j + n;
      c_test[i][j] = c[i][j];
    }
  }
  
  for(i=0;i<2;i++){
    for(j=0;j<3;j++){
      for(k=0;k<4;k++){
	d[i][j][k] = i*3*4 + j*4 + k + n;
	d_test[i][j][k] = d[i][j][k];
      }
    }
  }
  xmp_sync_all(&status);
}

void communicate_1(int me){
  int tmp;

  if(me == 1){   // get
    tmp = a;
    a = a:[2];
  }
  xmp_sync_all(&status);
  if(me == 1){  // put
    a:[2] = tmp;
  }
  
  if(me == 1){
    a_test = 101;
  }
  else if(me == 2){
    a_test = 1;
  }
  
  xmp_sync_all(&status);
}

void check_1(int me){
  if(a == a_test)
    printf("[%d] check_1 : PASS\n", me);
  else{
    printf("[%d] check_1 : fall\n[%d] a = %d (True value is %d)\n", me, me, a, a_test);
    return_val = 1;
  }
}

void communicate_2(int me){
  if(me == 2){
    long tmp = 99;
    b[0]:[1] = tmp;   // put
    xmp_sync_memory(&status);
    b[1] = b[0]:[1];  // get
  }
  
  if(me == 1){
    b_test[0] = 99;
  }
  else if(me == 2){
    b_test[1] = 99;
  }
  
  xmp_sync_all(&status);
}

void check_2(int me){
  int i, flag = TRUE;
  
  for(i=0; i<2; i++){
    if( b[i] != b_test[i] ){
      flag = FALSE;
      printf("[%d] b[%d] check_2 : fall\nb[%d] = %ld (True value is %ld)\n",
             me, i, i, b[i], b_test[i]);
    }
  }
  
  if(flag == TRUE)   printf("[%d] check_2 : PASS\n", me);
  else return_val = 1;
}

void communicate_3(int me){
  float tmp[2][3];

  tmp[0][0] = 9.1;
  xmp_sync_all(&status);
  if(me == 1){
    c[1][1]:[2] = tmp[0][0]; // put
	} 
  if(me == 2){
    c[1][2] = c[1][0]:[1];  // get
  }
  
  if(me == 2){
    c_test[1][1] = 9.1;
    c_test[1][2] = 3;
  }
  xmp_sync_all(&status);
}

void check_3(int me){
  int i, j, flag = TRUE;

  for(i=0; i<2; i++){
    for(j=0; j<3; j++){
      if( c[i][j] != c_test[i][j] ){
        flag = FALSE;
	printf("[%d] check_3 : c[%d][%d] = %f (True value is %f) : ERROR\n",
	       me, i, j, c[i][j], c_test[i][j]);
      }
    }
  }
  
  if(flag == TRUE)  printf("[%d] check_3 : PASS\n", me);
  else return_val = 1;
}

void communicate_4(int me){
  int dest = 2, src = 1;

  xmp_sync_all(&status);
  if(me == src){
    double tmp;
    tmp = d[0][1][2];
    d[0][1][2] = 3.14;
    d[1][2][3]:[dest] = d[0][1][2]; // put
    d[0][1][2] = tmp;
  }
  
  if(me == src){
    double tmp[1][2][3];
    tmp[0][1][2] = d[1][1][2]:[dest];
    d[1][1][2] = tmp[0][1][2];
  }
  
  if(me == dest){
    d_test[1][2][3] = 3.14;
  } 
  else if(me == src){
    d_test[1][1][2] = 118;
  }
  xmp_sync_all(&status);
}

void check_4(int me){
  int i, j, k, flag = TRUE;
  
  for(i=0; i<2; i++){
    for(j=0; j<3; j++){
      for(k=0; k<4; k++){
	if( d[i][j][k] != d_test[i][j][k] ){
	  flag = FALSE;
	  printf("[%d] check_4 : d[%d][%d][%d] = %f (True value is %f) : ERROR\n",
		 me, i, j, k, d[i][j][k], d_test[i][j][k]);
	}
      }
    }
  }
  
  if(flag == TRUE)  printf("[%d] check_4 : PASS\n", me);
  else return_val = 1;
}

int main(){
  int me = xmp_node_num();
  
  initialize_coarrays(me);
  
  communicate_1(me);
  check_1(me);
  
  communicate_2(me);
  check_2(me);
  
  communicate_3(me);
  check_3(me);
  
  communicate_4(me);
  check_4(me);

#pragma xmp barrier
#pragma xmp reduction(MAX:return_val)
  return return_val;
}
