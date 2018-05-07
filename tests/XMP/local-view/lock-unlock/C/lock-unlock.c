#include <xmp.h>
#include <stdlib.h>
#include <stdio.h>

xmp_lock_t lockobj:[*];
int a:[*];

int main(){
  if(xmpc_this_image() == 0)
    a = 1;
  else
    a = -1;

#pragma xmp lock(lockobj:[1])
  if(xmpc_this_image() == 0){
    a:[1] = a;
    xmp_sync_memory(NULL);
  }
#pragma xmp unlock(lockobj:[1])

  if(xmpc_this_image() == 0)
    printf("PASS\n");

  xmp_sync_all(NULL);
  
  return 0;
}
