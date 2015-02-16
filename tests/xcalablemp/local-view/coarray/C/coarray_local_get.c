#include <stdio.h>
#include "xmp.h"

#define TRUE 1
#define FALSE 0
int a[20];
#pragma xmp nodes p(3)
#pragma xmp coarray a:[*]

int main()
{
  int me, n1, n2, n3;
  int i, stat, check = FALSE;

  me = xmp_node_num();
  for (i = 0; i < 20; i++)
    a[i] = i*me;

  xmp_sync_all(&stat);

  n1 = a[10]:[1];
  n2 = a[10]:[2];
  n3 = a[10]:[3];

  xmp_sync_all(&stat);

  if(n1 == 10 && n2 == 20 && n3 == 30)
    check = TRUE;

#pragma xmp barrier
#pragma xmp reduction(MIN:check)

  if(check){
    if(me == 1)printf("PASS\n");
    return 0;
  }
  else{
    if(me == 1)printf("ERROR\n");
    return 1;
  }
}
