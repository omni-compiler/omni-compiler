#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>
#define XMP_FALSE 0
#define XMP_TRUE  1
#pragma xmp nodes p[*]
#pragma xmp template t[10]
#pragma xmp distribute t[block] onto p
struct test{
  int a[10];
};
#pragma xmp align test.a[i] with t[i]
#pragma xmp shadow test.a[1]
struct test c;

int main(){
#pragma xmp loop on t[i]
  for(int i=0;i<10;i++)
    c.a[i] = i;

  int flag = XMP_TRUE;
#pragma xmp reflect (c.a)

  if(xmpc_node_num() == 0){
    if(c.a[5] != 5)
      flag = XMP_FALSE;
  }
  else{
    if(c.a[4] != 4)
      flag = XMP_FALSE;
  }

#pragma xmp reduction(*:flag)
  if(flag == XMP_TRUE){
    if(xmpc_node_num() == 0) printf("OK\n");
  }
  else{
    if(xmpc_node_num() == 0) printf("Error !\n");
    exit(1);
  }

  return 0;
}
