#include <xmp.h>
#include <stdio.h>

int main(){
  int m[4] = {1, 2, 3, 4};
#pragma xmp nodes p(4)
#pragma xmp template t(0:9)
#pragma xmp distribute t(gblock(*)) onto p
#pragma xmp template_fix (gblock(m)) t

#pragma xmp loop on t(i)
  for(int i=0;i<10;i++)
    printf("[%d] %d\n", xmpc_node_num(), i);

  return 0;
}
