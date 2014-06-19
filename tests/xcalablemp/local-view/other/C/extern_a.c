#include <xmp.h>
#include <stdio.h>
#include <stdlib.h>
int a[10];
#pragma xmp nodes p(2)
#pragma xmp coarray a:[*]
extern void hoge(int i, int node, int value);

int main(){
  for(int i=0;i<10;i++)
    a[i] = 0;
  
  xmp_sync_all(NULL);
  if(xmp_node_num() == 1)
    hoge(3, 2, -9);
  
  xmp_sync_all(NULL);

  if(xmp_node_num() == 2){
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
