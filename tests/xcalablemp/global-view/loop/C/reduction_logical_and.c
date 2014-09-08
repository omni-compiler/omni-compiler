#include <xmp.h>
#include <stdio.h>  
#include <stdlib.h> 
#define TRUE  1
#define FALSE 0
int N=4,i,a[N], sa=FALSE, result=0;
#pragma xmp nodes p(4)
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(cyclic) onto p
#pragma xmp align a[i] with t(i)

int main(void)
{
#pragma xmp loop on t(i)
  for(i=0;i<N;i++)
    a[i] = FALSE;

#pragma xmp loop on t(i) reduction(&&:sa)
  for(i=0;i<N;i++)
    sa = a[i];    

  if(sa != FALSE)  result = -1;

#pragma xmp task on p(1)
  {
    a[0] = TRUE;
  }

#pragma xmp loop on t(i) reduction(&&:sa)
  for(i=0;i<N;i++)
    sa = a[i];

  if(sa != FALSE) result = -1;

#pragma xmp loop on t(i)
  for(i=0;i<N;i++)
    a[i] = TRUE;

#pragma xmp loop on t(i) reduction(&&:sa)
  for(i=0;i<N;i++)
    sa = a[i];

  if(sa != TRUE) result = -1;

#pragma xmp reduction(+:result)
#pragma xmp task on p(1)
  {
    if(result == 0){
      printf("PASS\n");
    }
    else{
      fprintf(stderr, "ERROR\n");
      exit(1);
    }
  }

  return 0;
}
