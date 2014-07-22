#include <xmp.h>
#include <stdio.h>
#include <stdlib.h>
#pragma xmp nodes p(4,4)
#pragma xmp template t(0:3,:)
#pragma xmp distribute t(cyclic,block) onto p
int i,j,N,s,result=0, **a;
#pragma xmp align a[i][j] with t(j,i)

int main(void)
{
  N = 1000;
#pragma xmp template_fix(block) t(0:N-1,0:N-1)
  for(i=0;i<N;i++)
    a[i]=(int *)malloc(sizeof(int) * N);
  
#pragma xmp loop (j,i) on t(j,i)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      a[i][j] = xmp_node_num();

  s=0;
#pragma xmp loop (j,i) on t(j,i) reduction(+:s)
  for(i=1;i<N+1;i++)
    for(j=1;j<N+1;j++)
      s = s+a[i][j];

  if(s != 450000)
    result = -1;

  for(i=0;i<1000;i++)
    free(a[i]); 
   return 0;
}
