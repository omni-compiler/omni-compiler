#include <xmp.h>
#include <stdio.h> 
#include <stdlib.h>
#define N 4
int random_array[N], ans=-1, val=-1, i, result=0;
#pragma xmp nodes p(4)
#pragma xmp template t(0:3)
#pragma xmp distribute t(block) onto p

int main(void)
{
  srand(0);
  for(i=0;i<N;i++){
    random_array[i] = rand();
    ans = ans & random_array[i];
  }
  
#pragma xmp loop on t(i)
  for(i=0;i<N;i++)
    val = random_array[i];

#pragma xmp reduction(&:val)

  if(val != ans)
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
