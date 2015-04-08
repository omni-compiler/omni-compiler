#include <stdio.h>
#include "xmp.h"
#define TRUE 1
#define FALSE 0
#define N 20
int a[N];
#pragma xmp nodes p(3)
#pragma xmp coarray a:[*]

int main()
{
  int me, n[3];
  int i, stat, check = FALSE;

  me = xmp_node_num();
  for (i = 0; i < 20; i++)
    a[i] = -1;

  n[0] = n[1] = n[2] = me * 100;  

  xmp_sync_all(&stat);

  a[0]:[me] = n[0];
  a[1:2:3]:[me] = n[1];
  
  xmp_sync_all(&stat);

  if(a[0] == me * 100 && a[1] == me * 100 && a[4] == me * 100)
    check = TRUE;

#pragma xmp barrier
#pragma xmp reduction(MIN:check)

  if(check){
    if(me == 1) printf("PASS\n");
    return 0;
  }
  else{
    if(me == 1) printf("ERROR\n");
    return 1;
  }
}
