#include <stdio.h>
#include <xmp.h>
#pragma xmp nodes p[*]

void hello(){
  printf("Hello on node p[%d]\n", xmp_node_num());
}
