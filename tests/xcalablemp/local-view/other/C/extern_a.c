#include <xmp.h>
#include <stdio.h>
#include <stdlib.h>
int a[10];
#pragma xmp coarray a:[*]
extern void hoge(int i, int node, int value);

int main(){
  for(int i=0;i<10;i++)
    a[i] = 0;
  
  xmp_sync_all(NULL);
  if(xmpc_this_image() == 0)
    hoge(3, 1, -9);
  
  xmp_sync_all(NULL);

  if(xmpc_this_image() == 1){
    if(a[3] == -9){
      printf("PASS\n");
    }
    else{
      fprintf(stderr, "ERROR\n");
      exit(1);
    }
  }
  return 0;
}
