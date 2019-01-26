#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>
#pragma xmp nodes p[*]
#pragma xmp template t[10]
#pragma xmp distribute t[block] onto p
struct b{
  int a[10];
};
#pragma xmp align b.a[i] with t[i]
struct b c;

int main(){
  int sum = 0;
#pragma xmp loop on t[i]
  for(int i=0;i<10;i++)
    c.a[i] = i;

#pragma xmp loop on t[i] reduction(+:sum)
  for(int i=0;i<10;i++)
    sum += c.a[i];

  if(sum != 45){
    if(xmpc_node_num() == 0) printf("Error !\n");
    exit(1);
  }
  else
    if(xmpc_node_num() == 0) printf("OK\n");
  
  return 0;
}
