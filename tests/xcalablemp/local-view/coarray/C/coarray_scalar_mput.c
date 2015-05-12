#define TRUE 1
#define FALSE 0
#include <stdio.h>
#include <xmp.h>
#pragma xmp nodes p(2)
#define N 10
long a[N], a_ans[N];
float b[N][N], b_ans[N][N];
double c[N][N][N], c_ans[N][N][N];
#pragma xmp coarray a,b,c : [*]
int status, return_val = 0;

void initialize_coarrays(int me)
{
  for(int i=0;i<N;i++)
    a[i] = a_ans[i] = 0;

  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      b[i][j] = b_ans[i][j] = 0;

  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      for(int k=0;k<N;k++)
	c[i][j][k] = c_ans[i][j][k] = 0;

  xmp_sync_all(&status);
}

void test_1(int me){

  if(me == 2){
    long tmp = 99;
    a[0:5]:[1] = tmp;   // put
    xmp_sync_memory(&status);
  }

  if(me == 1){
    for(int i=0;i<5;i++)
      a_ans[i] = (long)99;
  }
  
  xmp_sync_all(&status);
}

void check_1(int me){
  int flag = TRUE;
  
  for(int i=0;i<N;i++){
    if(a[i] != a_ans[i]){
      flag = FALSE;
      printf("[%d] a[%d] check_1 : fall\na[%d] = %ld (True value is %ld)\n",
             me, i, i, a[i], a_ans[i]);
    }
  }
  
  if(flag == TRUE && me == 2)   printf("check_1 : PASS\n");
  if(flag == FALSE) return_val = 1;
}

void test_2(int me){
  float tmp[2][3];
  tmp[0][1] = 9.1;

  xmp_sync_all(&status);

  if(me == 1)
    b[1:5:2][1]:[2] = tmp[0][1]; // put
  
  if(me == 2){
    for(int i=0;i<5;i++)
      b_ans[1+i*2][1] = 9.1;
  }

  xmp_sync_all(&status);
}

void check_2(int me){
  int flag = TRUE;

  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      if(b[i][j] != b_ans[i][j]){
        flag = FALSE;
	printf("[%d] check_2 : b[%d][%d] = %f (True value is %f) : ERROR\n",
	       me, i, j, b[i][j], b_ans[i][j]);
      }
    }
  }
  
  if(flag == TRUE && me == 2)  printf("check_2 : PASS\n");
  if(flag == FALSE) return_val = 1;
}

void test_3(int me){
  int dest = 1, src = 2;

  xmp_sync_all(&status);
  if(me == src){
    double tmp = 3.14;
    c[0][1][2] = tmp;
    c[1][2:3:3][3]:[dest] = c[0][1][2]; // put
    c[0][1][2] = 0;
  }
  
  if(me == dest){
    c_ans[1][2][3] = 3.14;
    c_ans[1][5][3] = 3.14;
    c_ans[1][8][3] = 3.14;
  } 

  xmp_sync_all(&status);
}

void check_3(int me){
  int flag = TRUE;
  
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      for(int k=0;k<N;k++){
	if(c[i][j][k] != c_ans[i][j][k]){
	  flag = FALSE;
	  printf("[%d] check_3 : c[%d][%d][%d] = %f (True value is %f) : ERROR\n",
		 me, i, j, k, c[i][j][k], c_ans[i][j][k]);
	}
      }
    }
  }
  
  if(flag == TRUE && me == 1)  printf("check_3 : PASS\n");
  if(flag == FALSE) return_val = 1;
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

#pragma xmp barrier
#pragma xmp reduction(MAX:return_val)
  return return_val;
}
