#include <xmp.h>
extern int a[10];
#pragma xmp coarray a:[*]

void hoge(int i, int node, int value)
{
  a[i]:[node] = value;
}
