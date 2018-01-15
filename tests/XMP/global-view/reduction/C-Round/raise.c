#include <xmp.h>
#include <stdio.h>
#include <stdlib.h>
static const int N=1000;
#pragma xmp nodes p(*)
int random_array[1000],ans_val=0,val=0;
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(block) onto p
int i,result=0;

int main(void)
{
  srand(0);
  for(i=0;i<N;i++)
    random_array[i] = rand();

  for(i=0;i<N;i++)
    ans_val = ans_val^random_array[i];

#pragma xmp loop on t(i)
  for(i=0;i<N;i++)
    val = val^random_array[i];
#pragma xmp reduction(^:val)

  if(val != ans_val)
    result = -1;

#pragma xmp reduction(+:result)
#pragma xmp task on p(1)
  {
    if(result == 0 ){
      printf("PASS\n");
    }
    else{
      fprintf(stderr, "ERROR\n");
      exit(1);
    }
  }

  return 0;
}
