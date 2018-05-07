#include <stdio.h>
#include <xmp.h>
#pragma xmp nodes p[*]

void hello_0(){
  printf("[%d]\n", xmp_node_num());
}

void hello_1(long a[3]){
  printf("[%d] %d %d %d\n", xmp_node_num(), a[0], a[1], a[2]);
}

void hello_2(long a[3], long b[3]){
  printf("[%d] %d %d %d : %d %d %d\n",
	 xmp_node_num(), a[0], a[1], a[2], b[0], b[1], b[2]);
}
