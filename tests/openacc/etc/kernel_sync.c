#include<openacc.h>

int main()
{
  int i = 0;
  int a[1000];
  int ret;
  
#pragma acc data create(a)
  {
#pragma acc parallel loop
    for(i=0;i<1000;i++){
      a[i] = i*i*i*i/7;
    }

    ret = acc_async_test(ACC_ASYNC_SYNC);
  }

  if(ret == 0){ //not completed
    return 1;
  }

  return 0;
}
