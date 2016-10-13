#include <xmp.h>
#include <stdbool.h>
#include <stdio.h>
int locked:[*];
int val = true;

int main()
{
  int iam = xmp_node_num();
  locked = true;
  
  if(iam == 1){
    xmp_sync_memory(NULL);
    xmp_atomic_define(locked:[2], false);
  }
  else if(iam == 2){
    val = true;
    while(val){
      xmp_atomic_ref(&val, locked);
    }
    xmp_sync_memory(NULL);
  }

  if(xmp_node_num() == 1)
    printf("PASS\n");
  
  return 0;
}
