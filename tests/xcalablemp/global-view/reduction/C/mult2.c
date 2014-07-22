#include <xmp.h>
#include <stdio.h>
#include <stdlib.h>   
static const int N=1000;
#pragma xmp nodes p(4,*)
#pragma xmp template t1(0:N-1,0:N-1)
#pragma xmp distribute t1(block,block) onto p
int a[N][N],sa=1;
int i,j,m,result=0;
#pragma xmp align a[i][j] with t1(j,i)

int main(void)
{
#pragma xmp loop (j,i) on  t1(j,i)
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      if((i==j)&&(i%100==0)){
	a[i][j] = 2;
      }else{
	a[i][j] = 1;
      }
    }
  }

#pragma xmp loop (j,i) on t1(j,i)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      sa = sa*a[i][j];
#pragma xmp reduction (*:sa) 

  if(sa != 1024)
    result = -1;  // ERROR

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
