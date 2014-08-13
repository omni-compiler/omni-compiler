#include <stdio.h>
#include <stdlib.h>     
#include "xmp.h"

#define N 120
int m[4] = { 8, 16, 32, 64 };
int NSIZE = N;

#pragma xmp nodes p(4)
#pragma xmp template t(:)
#pragma xmp distribute t(gblock(*)) onto p

int xmp_node_num();
void foo()
{
#pragma xmp template_fix (gblock(m)) t(0:NSIZE-1)
  int *a;
#pragma xmp align a[i] with t(i)
#pragma xmp shadow a[2]

  a = (int*)xmp_malloc(xmp_desc_of(a), NSIZE);

#pragma xmp loop on t(i)
  for (int i = 0; i < N; i++)
    a[i] = i;

#pragma xmp reflect (a) width (/periodic/1:1) async (100)
#pragma xmp wait_async (100)

#pragma xmp loop (i) on t(i)
  for (int i = 0; i < N; i++){
    if(a[i-1] != (i-1+N)%N){
      printf("ERROR Lower in %d (%06d) (%06d)\n", (int)i, a[i-1], (int)(i-1+N)%N);
      exit(1);
    }
    if(a[i+1] != (i+1+N)%N){
      printf("ERROR Upper in %d (%06d) (%06d)\n", (int)i, a[i+1], (int)(i+1+N)%N);
      exit(1);
    }
  }

#pragma xmp task on p(1)
  {
    printf("PASS\n");
  }
}

int main()
{
  foo();
  return 0;
}
