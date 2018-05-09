#include<xmp.h>
#include<stdio.h>
#include<stdlib.h>     
#pragma xmp nodes p(4,4)
#pragma xmp template t(0:999,0:999,:)
#pragma xmp distribute t(*,cyclic(3),cyclic(7)) onto p
int i,j,N,s, **a,result=0;
#pragma xmp align a[i][j] with t(*,j,i)

int main(void)
{
  N = 1000;

#pragma xmp template_fix(*,cyclic(3),block(7)) t(N,N,N)
  for(i=0;i<N;i++)
    a[i]=(int *)malloc(sizeof(int) * N);

#pragma xmp loop (j,i) on t(:,j,i)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      a[i][j] = xmp_node_num();

   s = 0;
#pragma xmp loop (j,i) on t(j,i) reduction(+:s)
   for(i=0;i<N;i++)
     for(j=0;j<N;j++)
       s += a[i][j];
   
   result = 0;
   if(s != 79300)
     result = -1;

#pragma xmp reduction(+:result)
#pragma xmp task on p(1,1)
   {
     if(result == 0){
       printf("PASS\n");
     }
     else{
       fprintf(stderr, "ERROR\n");
       exit(1);
     }
   }

   for(i=0;i<N;i++)
     free(a[i]);
 
   return 0;
}
