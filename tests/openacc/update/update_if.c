#include <stdio.h>
#define N 100


int main()
{
  int a = 0;
  int flag;
  
#pragma acc data copyin(a)
  {
    //true (const)    
#pragma acc kernels
    a = 1;
#pragma acc update host(a) if(1)
    if(a != 1) return 1;


    //false (const)
#pragma acc kernels
    a = 2;
#pragma acc update host(a) if(0)
    if(a != 1) return 2;

    
    //true (variable)
#pragma acc kernels
    a = 3;
    flag = -3;
#pragma acc update host(a) if(flag)
    if(a != 3) return 3;

    
    //false (variable)
#pragma acc kernels
    a = 4;
    flag = 0;
#pragma acc update host(a) if(flag)
    if(a != 3) return 4;
  }
  
  printf("PASS\n");
  return 0;
}
