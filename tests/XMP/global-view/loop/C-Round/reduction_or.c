#include <xmp.h>
#include <stdio.h>
#include <stdlib.h>
static const int N=10;
int random_array[10],ans_val=0;
int a[N],sa=0;
#pragma xmp nodes p(*)
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(cyclic) onto p
int i,k,result=0;
#pragma xmp align a[i] with t(i)

int main(void)
{
  srand(0);
  for(i=0;i<N;i++)
    random_array[i] = rand();
    
#pragma xmp loop (i) on t(i)
  for(i=0;i<N;i++)
    a[i] = random_array[i];

#pragma xmp loop (i) on t(i) reduction(|:ans_val)
  for(i=0;i<N;i++)
    ans_val = ans_val|a[i];

  for(i=0;i<N;i++)
    sa = sa | random_array[i];
  
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
