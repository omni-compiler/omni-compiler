#include <stdio.h>
#include "acc_func.h"

int main()
{
  int r = acc_func();
  
  if(r == 0){
    printf("PASS\n");
  }
  return r;
}
