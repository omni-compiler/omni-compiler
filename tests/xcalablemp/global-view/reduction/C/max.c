#include <xmp.h>
#include <stdio.h> 
#include <stdlib.h>   
static const int N=1000;
int random_array[1000],ans_val=0,val=0,result=0,i;
#pragma xmp nodes p(*)
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(block) onto p

int main(void){
  srand(0);
  for(i=0;i<N;i++)
    random_array[i] = rand();
   
#pragma xmp loop on t(i)
    for(i=0;i<N;i++)
      if(random_array[i]>val)
	val = random_array[i];

#pragma xmp reduction(max:val)

    ans_val = 0;
    for(i=0;i<N;i++)
      if(random_array[i]>ans_val)
	ans_val = random_array[i];

    if(val != ans_val)
      result = -1; // NG

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

