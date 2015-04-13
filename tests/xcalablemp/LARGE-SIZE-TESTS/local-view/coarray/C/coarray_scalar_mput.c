#define TRUE 1
#define FALSE 0
#include <stdio.h>
#include <xmp.h>
#pragma xmp nodes p(2)
#define N 10
long   a[N][N][N][N]:[*],          a_ans[N][N][N][N];
float  b[N][N][N][N][N]:[*],       b_ans[N][N][N][N][N];
double c[N][N][N][N][N][N]:[*],    c_ans[N][N][N][N][N][N];
long   d[N][N][N][N][N][N][N]:[*], d_ans[N][N][N][N][N][N][N];
int status, return_val = 0;

void initialize_coarrays(int me)
{
  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      for(int k=0;k<N;k++)
        for(int x=0;x<N;x++)
	  a[i][j][k][x] = a_ans[i][j][k][x] = 0;

  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      for(int k=0;k<N;k++)
	for(int x=0;x<N;x++)
	  for(int y=0;y<N;y++)
	    b[i][j][k][x][y] = b_ans[i][j][k][x][y] = 0;

  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      for(int k=0;k<N;k++)
	for(int x=0;x<N;x++)
	  for(int y=0;y<N;y++)
	    for(int z=0;z<N;z++)
	      c[i][j][k][x][y][z] = c_ans[i][j][k][x][y][z] = 0;

  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      for(int k=0;k<N;k++)
        for(int x=0;x<N;x++)
          for(int y=0;y<N;y++)
            for(int z=0;z<N;z++)
	      for(int m=0;m<N;m++)
		d[i][j][k][x][y][z][m] = d_ans[i][j][k][x][y][z][m] = 0;

  xmp_sync_all(&status);
}

void test_4(int me)
{
  if(me == 2){
    long tmp = 99;
    a[1][1:5][:][2]:[1] = tmp;   // put
    xmp_sync_memory(&status);
  }

  if(me == 1){
    for(int j=1;j<6;j++)
      for(int k=0;k<N;k++)
	a_ans[1][j][k][2] = (long)99;
  }
  
  xmp_sync_all(&status);
}

void check_4(int me)
{
  int flag = TRUE;
  
  if(me == 1)
    for(int i=0;i<N;i++){
      for(int j=0;j<N;j++){
	for(int k=0;k<N;k++){
	  for(int x=0;x<N;x++){
	    if(a[i][j][k][x] != a_ans[i][j][k][x]){
	      flag = FALSE;
	      printf("[%d] a[%d][%d][%d][%d] check_1 : fall %ld (True value is %ld)\n",
		     me, i, j, k, x, a[i][j][k][x], a_ans[i][j][k][x]);
	    }
	  }
	}
      }
    }

  if(flag == TRUE && me == 1)   printf("check_4 : PASS\n");
  if(flag == FALSE) return_val = 1;
}

void test_5(int me)
{
  float tmp[2][3];
  tmp[0][1] = 9.1;

  xmp_sync_all(&status);

  if(me == 1)
    b[1:4:2][1][:][:][:]:[2] = tmp[0][1];
  
  if(me == 2){
    for(int k=0;k<N;k++)
      for(int x=0;x<N;x++)
	for(int y=0;y<N;y++){
	  b_ans[1][1][k][x][y] = 9.1;
	  b_ans[3][1][k][x][y] = 9.1;
	  b_ans[5][1][k][x][y] = 9.1;
	  b_ans[7][1][k][x][y] = 9.1;
	}
  }

  xmp_sync_all(&status);
}

void check_5(int me){
  int flag = TRUE;
  
  if(me == 2)
    for(int i=0;i<N;i++)
      for(int j=0;j<N;j++)
	for(int k=0;k<N;k++)
	  for(int x=0;x<N;x++)
	    for(int y=0;y<N;y++)
	      if(b[i][j][k][x][y] != b_ans[i][j][k][x][y]){
		flag = FALSE;
		printf("check_5 : b[%d][%d][%d][%d][%d] = %f (True value is %f) : ERROR\n",
		       i, j, k, x, y, b[i][j][k][x][y], b_ans[i][j][k][x][y]);
	      }
  
  if(flag == TRUE && me == 2)  printf("check_2 : PASS\n");
  if(flag == FALSE) return_val = 1;
}

void test_6(int me)
{
  xmp_sync_all(&status);
  if(me == 2){
    double tmp = 3.14;
    c[0][1][2][0][1][2] = tmp;
    c[1][2:3:3][3][2][2][2]:[1] = c[0][1][2][0][1][2];
    c[0][1][2][0][1][2] = 0;
}

  if(me == 1){
    c_ans[1][2][3][2][2][2] = 3.14;
    c_ans[1][5][3][2][2][2] = 3.14;
    c_ans[1][8][3][2][2][2] = 3.14;
  }

  xmp_sync_all(&status);
}

void check_6(int me)
{
  int flag = TRUE;
  
  if(me == 1)
    for(int i=0;i<N;i++)
      for(int j=0;j<N;j++)
	for(int k=0;k<N;k++)
	  for(int x=0;x<N;x++)
	    for(int y=0;y<N;y++)
	      for(int z=0;z<N;z++)
		if(c[i][j][k][x][y][z] != c_ans[i][j][k][x][y][z]){
		  flag = FALSE;
		  printf("check_6 : c[%d][%d][%d][%d][%d][%d] = %f (True value is %f) : ERROR\n",
			 i, j, k, x, y, z, c[i][j][k][x][y][z], c_ans[i][j][k][x][y][z]);
		}
  
  if(flag == TRUE && me == 1)  printf("check_6 : PASS\n");
  if(flag == FALSE) return_val = 1;
}

void test_7(int me)
{
  xmp_sync_all(&status);
  if(me == 2){
    long u = 3;
    d[1:2][2:3:3][3][2][:][:][:]:[1] = u;
  }

  if(me == 1){
    for(int y=0;y<N;y++)
      for(int z=0;z<N;z++)
	for(int m=0;m<N;m++){
	  d_ans[1][2][3][2][y][z][m] = 3;
	  d_ans[1][5][3][2][y][z][m] = 3;
	  d_ans[1][8][3][2][y][z][m] = 3;
          d_ans[2][2][3][2][y][z][m] = 3;
          d_ans[2][5][3][2][y][z][m] = 3;
          d_ans[2][8][3][2][y][z][m] = 3;
	}
  }

  xmp_sync_all(&status);
}

void check_7(int me)
{
  int flag = TRUE;

  if(me == 1)
    for(int i=0;i<N;i++)
      for(int j=0;j<N;j++)
        for(int k=0;k<N;k++)
          for(int x=0;x<N;x++)
            for(int y=0;y<N;y++)
              for(int z=0;z<N;z++)
		for(int m=0;m<N;m++)
		  if(d[i][j][k][x][y][z][m] != d_ans[i][j][k][x][y][z][m]){
		    flag = FALSE;
		    printf("check_7 : d[%d][%d][%d][%d][%d][%d][%d] = %ld (True value is %ld) : ERROR\n",
			   i, j, k, x, y, z, m, d[i][j][k][x][y][z][m], d_ans[i][j][k][x][y][z][m]);
                }
  
  if(flag == TRUE && me == 1)  printf("check_7 : PASS\n");
  if(flag == FALSE) return_val = 1;
}

int main(){
  int me = xmp_node_num();
  
  initialize_coarrays(me);
  
  test_4(me); check_4(me);
  test_5(me); check_5(me);
  test_6(me); check_6(me);
  test_7(me); check_7(me);

#pragma xmp barrier
#pragma xmp reduction(MAX:return_val)
  return return_val;
}
