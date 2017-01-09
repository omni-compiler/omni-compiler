#include <xmp.h>
#include <stdio.h>
#include <stdlib.h>
int atom:[*];
int tmp:[*];
int main()
{
  tmp = xmp_node_num();
  atom = xmp_node_num();
  int value = -1;

  xmp_sync_all(NULL);
  if(xmp_node_num() == 2){
    atom:[1] = tmp;
    xmp_atomic_ref(&value, atom:[1]);
    if(value != 2){
      printf("ERROR!\n");
      exit(1);
    }
    else{
      printf("PASS\n");
    }
  }
  xmp_sync_all(NULL);

  return 0;
}
