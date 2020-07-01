#include <xmp.h>
#include <stdio.h> 
#include <stdlib.h>   
static const int N=1000;
int random_array[1000],ans_val=RAND_MAX,val=RAND_MAX,result = 0,i;
#pragma xmp nodes p[*]
#pragma xmp template t[N]
#pragma xmp distribute t[block] onto p

int main(void)
{
  srand(0);
  for(i=0;i<N;i++)
    random_array[i] = rand();
  
#pragma xmp loop on t[i]
  for(i=0;i<N;i++)
    if(random_array[i]<val)
      val = random_array[i];
  
#pragma xmp reduction(min:val)

  for(i=0;i<N;i++)
    if(random_array[i]<ans_val)
      ans_val = random_array[i];

  if(val != ans_val)
    result = -1;

#pragma xmp reduction(+:result)
#pragma xmp task on p[0]
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
