#define TRUE 1
#define FALSE 0
#include <stdio.h>
#include <xmp.h>
#pragma xmp nodes p[2]
#define N 10
long a[N];
float b[N][N];
double c[N][N][N];
#pragma xmp coarray a,b,c : [*]
int return_val = 0;

void initialize_coarrays(int me)
{
  for(int i=0;i<N;i++)
    a[i] = i + (me+1)*100;

  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      b[i][j] = i*N + j + (me+1)*100;

  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      for(int k=0;k<N;k++)
	c[i][j][k] = i*N*N + j*N + k + (me+1)*100;

  xmp_sync_all(NULL);
}

void test_1(int me)
{
  if(me == 1){
    long tmp;
    tmp = a[1]:[0];
    xmp_sync_memory(NULL);

    if(tmp == a[1]-100)
      printf("check_1 : PASS\n");
    else{
      printf("check_1 : ERROR\n");
      return_val = 1;
    }
  }

  xmp_sync_all(NULL);
}

void test_2(int me)
{
  if(me == 0){
    float tmp[2][3];
    tmp[0:2][0] = b[1:2:2][1]:[1];
  
    if(tmp[0][0] == b[1][1]+100 && tmp[1][0] == b[3][1]+100){
      printf("check_2 : PASS\n");
    }
    else{
      printf("check_2 : ERROR\n");
      return_val = 1;
    }
  }

  xmp_sync_all(NULL);
}

void test_3(int me){
  int dest = 0, src = 1;

  xmp_sync_all(NULL);
  if(me == src){
    c[0][1:3:3][2] = c[1][2:3:2][3]:[dest];

    if(c[0][1][2] == c[1][2][3]-100 &&
       c[0][4][2] == c[1][4][3]-100 &&
       c[0][7][2] == c[1][6][3]-100){
      printf("check_3 : PASS\n");
    }
    else{
      printf("check_3 : ERROR\n");
      return_val = 1;
    }
  } 

  xmp_sync_all(NULL);
}

int main(){
  int me = xmpc_this_image();
  
  initialize_coarrays(me);
  
  test_1(me);
  test_2(me);
  test_3(me);

#pragma xmp barrier
#pragma xmp reduction(MAX:return_val)
  return return_val;
}
