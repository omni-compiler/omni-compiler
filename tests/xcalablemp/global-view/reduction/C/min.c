#include<xmp.h>
#include<stdio.h> 
#include<stdlib.h>   
static const int N=1000;
int random_array[1000], ans_val, val, result = 0, i, k;
#pragma xmp nodes p(*)
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(block) onto p

int main(void)
{
  for(k=114;k<10001;k=k+17){
    random_array[0]=k;
    for(i=1;i<N;i++){
      random_array[i]=(random_array[i-1]*random_array[i-1])%100000000;
      random_array[i]=(random_array[i]-((random_array[i]%100)/100))%10000;
    }
   
    val = 2147483647;
#pragma xmp loop on t(i)
    for(i=0;i<N;i++){
      if(random_array[i]<val)
	val = random_array[i];
    }
#pragma xmp reduction(min:val)
    ans_val = 2147483647;
    for(i=0;i<N;i++){
      if(random_array[i]<ans_val)
	ans_val = random_array[i];
    }
    if(val != ans_val){
      result = -1;
    }
  }

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
