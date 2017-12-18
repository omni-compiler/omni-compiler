#include <stdio.h>
#include <xmp.h>
#pragma xmp nodes p[*]

void hello(long a[3]){
  printf("[%d] %d %d %d\n", xmp_node_num(), a[0], a[1], a[2]);
}
