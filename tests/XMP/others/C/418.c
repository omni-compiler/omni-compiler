#include <stdio.h>
#include <stdlib.h>

#define N 10
#pragma xmp nodes p[2]
#pragma xmp template t[N]
#pragma xmp distribute t(block) onto p
int a[N];
#pragma xmp align a[i] with t(i)

int main()
{
#pragma xmp loop on t[i]
  for(int i=0;i<N;i++)
    a[i] = i;
  
  int b = 0;
#pragma xmp loop on t[i] reduction(+:b)
#pragma omp parallel for  reduction(+:b)
  for(int i=0;i<N;i++)
    b += a[i];

  if(b == 45)
    printf("PASS\n");
  else{
    printf("ERROR\n");
    exit(1);
  }
  return 0;
}
