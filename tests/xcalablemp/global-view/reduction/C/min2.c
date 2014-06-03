#include <xmp.h>
#include <stdio.h> 
#include <stdlib.h>   
static const int N=1000;
int random_array[1000000],ans_val=RAND_MAX;
int a[1000][1000],sa=RAND_MAX;
double b[1000][1000],sb=RAND_MAX;
float c[1000][1000],sc=RAND_MAX;
#pragma xmp nodes p(4,*)
#pragma xmp template t(0:N-1,0:N-1)
#pragma xmp distribute t(block,block) onto p
int i,k,j,l,result=0;

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
      b[i][j] = (double)random_array[l];
      c[i][j] = (float)random_array[l];
    }
  }

#pragma xmp loop (j,i) on t(j,i)
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      if(a[i][j]<sa){
	sa = a[i][j];
      }
      if(b[i][j]<sb){
	sb = b[i][j];
      }
      if(c[i][j]<sc){
	sc = c[i][j];
      }
    }
  }
#pragma xmp reduction(min:sa,sb,sc) 

  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      l = j*N+i;
      if(random_array[l]<ans_val){
	ans_val = random_array[l];
      }
    }
  }

  if( sa != ans_val || sb != (double)ans_val || sc != (float)ans_val)
    result = -1;  // NG

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
