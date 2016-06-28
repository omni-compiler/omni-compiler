#include <openacc.h>
#include "acc_func.h"

int acc_func()
{
  int i;

#pragma acc parallel loop
  for(i=0;i<100;i++){
    int j;
    j = i;
  }

  acc_shutdown(acc_device_nvidia);

  return 0;
}
