#include <xmp.h>
#include <stdlib.h>
#include <stdio.h>

xmp_lock_t lockobj:[*];
int a:[*];

int main(){
  if(xmp_node_num() == 1)
    a = 1;
  else
    a = -1;

#pragma xmp lock(lockobj:[2])
  if(xmp_node_num() == 1){
    a:[2] = a;
    xmp_sync_memory(NULL);
  }
#pragma xmp unlock(lockobj:[2])

  if(xmp_node_num() == 1)
    printf("PASS\n");

  return 0;
}
