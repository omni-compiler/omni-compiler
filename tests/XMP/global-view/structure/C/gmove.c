#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>
#define XMP_FALSE 0
#define XMP_TRUE  1
#pragma xmp nodes p[2]
#pragma xmp template t[10]
#pragma xmp distribute t[block] onto p
struct b{
  int a[10];
};
int a[10];
#pragma xmp align b.a[i] with t[i]
#pragma xmp align a[i] with t[i]
struct b c;

int main(){
#pragma xmp loop on t[i]
  for(int i=0;i<10;i++){
    c.a[i] = i;
    a[i] = i + 100;
  }

#pragma xmp gmove
  a[0:3] = c.a[5:3];

  int flag = XMP_FALSE;
  if(xmpc_node_num() == 0){
    if(a[0] == 5 && a[1] == 6 && a[2] == 7 && a[3] == 103 && a[4] == 104)
      flag = XMP_TRUE;
  }
  
#pragma xmp reduction(+:flag)
    if(flag == XMP_TRUE){
      if(xmpc_node_num() == 0) printf("OK\n");
    }
    else{
      if(xmpc_node_num() == 0) printf("Error !\n");
      exit(1);
    }

  return 0;
}
