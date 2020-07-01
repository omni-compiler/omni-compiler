#include <stdio.h>
#include <xmp.h>
#define ARRAY_SIZE 4
#define TRUE 1
#define FALSE 0
#pragma xmp nodes p[2][2]
#pragma xmp template t[4][4]
#pragma xmp distribute t[block][block] onto p
int a[ARRAY_SIZE][ARRAY_SIZE];
#pragma xmp align a[i][j] with t[i][j]
#pragma xmp shadow a[1][1]

void error_msg(int i, int j, int top, int bottom, int left, int right){
  printf("(Error) Node %d a[%d][%d] = %d, top = %d, bottom = %d, left = %d, right = %d\n", 
	  xmp_node_num(), i, j, a[i][j], top, bottom, left, right);
}

int verify(int i, int j, int top, int bottom, int left, int right){
  int flag = TRUE;

  if(top != a[i][j]-4 || bottom != a[i][j]+4 || left != a[i][j]-1 || right != a[i][j]+1){
    error_msg(i, j, top, bottom, left, right);
    flag = FALSE;
  }

  return flag;
}

int main(void){
  int i, j, top, bottom, left, right;
  int flag = TRUE;

  /* Initialise array */
#pragma xmp loop on t[i][j]
  for(i=0; i<ARRAY_SIZE; i++)
    for(j=0; j<ARRAY_SIZE; j++)
      a[i][j] = i * ARRAY_SIZE + j;

#pragma xmp reflect (a)

#pragma xmp loop on t[i][j]
  for(i=0; i<ARRAY_SIZE; i++){
    for(j=0; j<ARRAY_SIZE; j++){
      
      if(i==0)
	top = (i - 1) * ARRAY_SIZE + j;
      else
      	top = a[i-1][j];

      if(ARRAY_SIZE-1 == i)
	bottom = (i + 1) * ARRAY_SIZE + j;
      else
	bottom = a[i+1][j];
      
      if(0==j)
	left = i * ARRAY_SIZE + j - 1;
      else
	left = a[i][j-1];
      
      if(ARRAY_SIZE-1 == j)
      	right = i * ARRAY_SIZE + j + 1;
      else
	right = a[i][j+1];

      if(!verify(i, j, top, bottom, left, right)){
	flag = FALSE;
      }
    }
  }

#pragma xmp reduction(MIN:flag)
#pragma xmp task on p[0][0]
  {
    if(flag) printf("PASS\n");
  }
  if(flag)
    return 0;
  else
    return 1;
}
