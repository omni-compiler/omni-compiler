#define NMAX 8
#include <stdio.h>
#include <stdlib.h>

int n=NMAX;
int a[n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n)
#pragma xmp distribute tx(block) onto p
#pragma xmp align a[i] with tx(i)

int main(){

int i, ierr=0;
#pragma xmp loop (i) on tx(i)
  for(i=0; i<n; i++){
    a[i]=i;
  }

#pragma xmp loop (i) on tx(i)
  for(i=0; i<n; i++){
    ierr = ierr + abs(a[i]-i);
  }

#pragma xmp reduction (MAX:ierr)

#pragma xmp task on p(1)
{
  if ( ierr == 0 ) {
     printf("PASS\n");
  }else{
     printf("ERROR\n");
     exit(1);
  }
}
  return 0;
}
