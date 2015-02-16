#include <stdio.h>
#include <stdlib.h>     
#include "xmp.h"

#define N 120
int m[4] = { 8, 16, 32, 64 };
int NSIZE = N;

#pragma xmp nodes p(4,*)
#pragma xmp template t(:,:)
#pragma xmp distribute t(gblock(*),block) onto p

void foo()
{
#pragma xmp template_fix (gblock(m),block) t(0:NSIZE-1,0:NSIZE-1)

  int (*a)[NSIZE];
#pragma xmp align a[i][j] with t(i,j)
#pragma xmp shadow a[2][1]

  a = (int (*)[NSIZE])xmp_malloc(xmp_desc_of(a), NSIZE, NSIZE);

#pragma xmp loop (i,j) on t(i,j)
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      a[i][j] = i * 1000 + j;
    }
  }

#pragma xmp reflect (a) width (/periodic/1:1,/periodic/1:1) async (100)
#pragma xmp wait_async (100)

#pragma xmp loop (i,j) on t(i,j)
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      if (a[i-1][j] != (i - 1 + N) % N * 1000 + j){
	printf("ERROR North in (%d, %d) (%06d) (%06d)\n", i, j, a[i-1][j], (i - 1 + N) % N * 1000 + j);
	exit(1);
      }
      if (a[i+1][j] != (i + 1 + N) % N * 1000 + j){
	printf("ERROR South in (%d, %d) (%06d) (%06d)\n", i, j, a[i+1][j], (i + 1 + N) % N * 1000 + j);
	exit(1);
      }
      if (a[i][j-1] != i * 1000 + (j - 1 + N) % N){
	printf("ERROR West in (%d, %d) (%06d) (%06d)\n", i, j, a[i][j-1], i * 1000 + (j - 1 + N) % N);
	exit(1);
      }
      if (a[i][j+1] != i * 1000 + (j + 1 + N) % N){
	printf("ERROR East in (%d, %d) (%06d) (%06d)\n", i, j, a[i][j+1], i * 1000 + (j + 1 + N) % N);
	exit(1);
      }
    }
  }

#pragma xmp task on p(1,1)
  {
    printf("PASS\n");
  }
}

int main()
{
  foo();
  return 0;
}
