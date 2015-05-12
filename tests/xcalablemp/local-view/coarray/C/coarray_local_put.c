#include <stdio.h>
#include <stdlib.h>
#define I 10
#define J 20
#define K 30
#define TRUE 1
#define FALSE 0
int a[I]:[*], b[I][J]:[*], c[I][J][K]:[*];
int a_normal[I], b_normal[I][J], c_normal[I][J][K];
#pragma xmp nodes p(1)

void test_ok(int flag)
{
  if(flag){
    printf("PASS\n");
  }
  else{
    printf("ERROR\n");
    exit(1);
  }
}

void initialize()
{
  for(int i=0;i<I;i++){
    a[i] = a_normal[i] = i;
    for(int j=0;j<J;j++){
      b[i][j] = b_normal[i][j] = (i * I) + j;
      for(int k=0;k<J;k++){
	c[i][j][k] = c_normal[i][j][k] = (i * I * J) + (j * J) + k;
      }
    }
  }  
}

int scalar_put()
{
  a[0]:[1]            = a_normal[0];
  a[3]:[1]            = b[1][2];
  a[4:2:2]:[1]        = b[2][2];
  b[0][3]:[1]         = a_normal[0];
  b[3][2]:[1]         = c[1][2][2];
  b[4:2:2][2:2:4]:[1] = c[2][2][1];

  int flag = TRUE;
  if(a[0] != a_normal[0]) flag = FALSE;
  if(a[3] != b[1][2]) flag = FALSE;
  if(a[4] != b[2][2] || a[6] != b[2][2]) flag = FALSE;
  if(b[0][3] != a_normal[0]) flag = FALSE;
  if(b[3][2] != c[1][2][2]) flag = FALSE;
  if(b[4][2] != c[2][2][1] || b[4][6] != c[2][2][1] ||
     b[6][2] != c[2][2][1] || b[6][6] != c[2][2][1]) flag = FALSE;

  return flag;
}

int vector_put()
{
  a[0:5]:[1]      = a_normal[2:5];
  b[0:4][3]:[1]   = a[5:4];
  b[3][2:2:2]:[1] = c[1:2:3][2][2];

  int flag = TRUE;
  for(int i=0;i<5;i++)
    if(a[i] != a_normal[2+i])
      flag = FALSE;

  for(int i=0;i<4;i++)
    if(b[i][3] != a[5+i])
      flag = FALSE;

  if(b[3][2] != c[1][2][2] || b[3][4] != c[4][2][2])
    flag = FALSE;

  return flag;
}

int main()
{
  initialize();
  test_ok(scalar_put());

  initialize();
  test_ok(vector_put());

  return 0;
}
