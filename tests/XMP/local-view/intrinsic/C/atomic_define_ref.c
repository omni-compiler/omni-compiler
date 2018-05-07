#include <xmp.h>
#include <stdbool.h>
#include <stdio.h>
int locked:[*];
int val = true;

int main()
{
  int iam = xmpc_this_image();
  locked = true;
  
  if(iam == 0){
    xmp_sync_memory(NULL);
    xmp_atomic_define(locked:[1], false);
  }
  else if(iam == 1){
    val = true;
    while(val){
      xmp_atomic_ref(&val, locked);
    }
    xmp_sync_memory(NULL);
  }

  if(xmpc_this_image() == 0)
    printf("PASS\n");
  
  return 0;
}
