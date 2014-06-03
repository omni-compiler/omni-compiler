#include<xmp.h>
#include<stdio.h>  
#include<stdlib.h> 
int N=1000,i,j,k,l,result=0;
int random_array[1000],ans_val=-1,val,a[N],sa=-1;
#pragma xmp nodes p(*)
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(cyclic) onto p
#pragma xmp align a[i] with t(i)

int main(void)
{
  srand(0);
  for(i=0;i<N;i++)
    random_array[i] = rand();
  
#pragma xmp loop on t(i)
  for(i=0;i<N;i++)
    a[i] = random_array[i];
  
  for(i=0;i<N;i++)
    ans_val = ans_val & random_array[i];

#pragma xmp loop on t(i) reduction(&:sa)
  for(i=0;i<N;i++)
    sa = sa & a[i];    

  if(sa != ans_val)
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
   return 0;
}
