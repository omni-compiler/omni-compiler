#include <stdio.h>
#include <stdlib.h>
#define N 128

int main()
{
  int i, a[N];

#pragma acc data copy(a)
  {
#pragma acc parallel loop if(1)
    for(i=0;i<N;i++){
      a[i] = 13;
    }
  }
  for(i=0;i<N;i++){
    if(a[i] != 13){
      exit(1);
    }
  }


  //test if(0)
#pragma acc data copy(a)
  {
#pragma acc kernels if(0)
    for (i=0;i<N;i++){
      a[i] = i*31;
    }

    for(i=0;i<N;i++){
      if(a[i] != i * 31){
	exit(2);
      }
    }
  }


  //test if(variable != 0)  
  int flag = 2;
#pragma acc data copy(a)
  {
#pragma acc kernels if(flag)
    for (i=0;i<N;i++){
      a[i] = i*17;
    }

    for(i=0;i<N;i++){
      if(a[i] != 13){
	exit(3);
      }
    }
  }

  for(i=0;i<N;i++){
    if(a[i] != i * 17){
      exit(4);
    }
  }

  //test if(variable == 0)  
  flag = 0;
#pragma acc data copy(a)
  {
#pragma acc kernels if(flag)
    for (i=0;i<N;i++){
      a[i] = i*19;
    }

    for(i=0;i<N;i++){
      if(a[i] != i * 19){
	exit(5);
      }
    }
  }

  for(i=0;i<N;i++){
    if(a[i] != i * 17){
      exit(6);
    }
  }


  printf("PASS\n");
  return 0;
}

