#include <stdio.h>

int func(int *a){
  int i,sum = 0;
#pragma acc data  present(a[0:10])
  {
    //#pragma acc host_data use_device(a)
    //  printf("func(a=%p)\n", a);
#pragma acc parallel loop reduction(+:sum)
  for(i=0;i<10;i++){
    sum += a[i];
  }
  }
  return sum;
}

int main()
{
  int i, a[20];

#pragma acc data create(a)
  {
    //#pragma acc host_data use_device(a)
    //    printf("func(a=%p)\n", a);

#pragma acc parallel loop 
    for(i=0;i<20;i++){
      a[i] = i;
    }


    int result = func(&a[5]);
    if(result != 95){
      printf("FAILED\n");
      return 1;
    }
  }
  printf("PASS\n");
  return 0;
}
