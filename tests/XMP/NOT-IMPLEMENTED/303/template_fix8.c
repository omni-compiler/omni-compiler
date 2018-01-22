#include <xmp.h>
#include <stdio.h>
#include <stdlib.h>
#pragma xmp nodes p(4,4)
#pragma xmp template t(:,:)
#pragma xmp distribute t(gblock(*),gblock(*)) onto p
static const int N=100;
int *m1,*m2, i,j, s,remain, **a, result=0;
#pragma xmp align a[i][j] with t(j,i)

int main(void)
{
  m1 = (int *)malloc(sizeof(int) * 4);
  m2 = (int *)malloc(sizeof(int) * 4);
  remain = N;

  for(i=0;i<3;i++){
    m1[i]=remain/2;
    remain =remain-m1[i];
  }
  m1[3]=remain;
  remain =N;
  for(i=0;i<3;i++){
    m2[i]=remain/3;
    remain=remain-m2[i];
  }
  m2[3]=remain;

#pragma xmp template_fix(gblock(m1),gblock(m2)) t(0:N-1,0:N-1)
  for(i=0;i<N;i++)
    a[i] = (int *)malloc(sizeof(int) * N);

#pragma xmp loop (j,i) on t(j,i)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      a[i][j]=xmp_node_num();

   s = 0;
#pragma xmp loop (j,i) on t(j,i) reduction(+:s)
  for(i=1;i<N+1;i++)
    for(j=1;j<N+1;j++)
      s += a[i][j];

  result = 0;
  if(s != 75600)
    result = -1;

#pragma xmp reduction(+:result)
#pragma xmp task on p(1)
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
     
  free(m1);
  free(m2);
  return 0;
}
      
         
      
   
   
