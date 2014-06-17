#include <xmp.h>
#include <stdio.h> 
#include <stdlib.h>
#define FALSE 0
#define TRUE 1
int val, i, result=0;
#pragma xmp nodes p(4)

int main(void)
{
  val = FALSE;
#pragma xmp reduction(&&:val)
  if(FALSE != val)
    result = -1;

#pragma xmp task on p(1)
  {
    val = TRUE;
  }

#pragma xmp reduction(&&:val)
  if(FALSE != val)
    result = -1;

  val = TRUE;
#pragma xmp reduction(&&:val)
  if(TRUE != val)
    result = -1;

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
