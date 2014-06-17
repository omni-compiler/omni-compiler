#include <xmp.h>
#include <stdio.h>  
#include <stdlib.h> 
static const int N=100;
int random_array[10000],ans_val=0;
int a[N][N],sa=0;
#pragma xmp nodes p(4,*)
#pragma xmp template t(0:N-1,0:N-1)
#pragma xmp distribute t(cyclic,cyclic) onto p
int i,j,k,l,result=0;
#pragma xmp align a[i][j] with t(j,i)

int main(void)
{
  srand(0);
  for(i=0;i<N*N;i++)
    random_array[i] = rand();

#pragma xmp loop (j,i) on t(j,i)
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      l = j*N+i;
      a[i][j] = random_array[l];
    }
  }
    
  for(i=0;i<N*N;i++)
    ans_val = ans_val | random_array[i];

#pragma xmp loop (j,i) on t(j,i) reduction(|:sa)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      sa = sa|a[i][j];    

  if(sa != ans_val)
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

  return 0;
}
