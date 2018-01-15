#include <xmp.h>
extern int a[10];
#pragma xmp coarray a:[*]

void hoge(int i, int image, int value)
{
  a[i]:[image] = value;
}
