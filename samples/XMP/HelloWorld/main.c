#include <stdio.h>
#include <xmp.h>
#pragma xmp nodes p[*]

int main()
{
  printf("node num = %d, Hello World\n", xmpc_node_num());
  return 0;
}
