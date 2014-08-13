#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>
#define NUM 1000
int status;
#pragma xmp nodes p(4)

int main()
{
  int id = xmp_node_num();

  if(id == 1){
    for(int i=0;i<NUM;i++){
#pragma xmp post (p(1), 2)
#pragma xmp post (p(2), 9)
#pragma xmp post (p(3), 8)
#pragma xmp post (p(4), 0)
    }
  }
  else if(id == 2){
    for(int i=0;i<NUM;i++){
#pragma xmp post (p(1), 9)
#pragma xmp post (p(2), 2)
#pragma xmp post (p(3), 1)
#pragma xmp post (p(4), 4)
    }
  }
  else if(id == 3){
    for(int i=0;i<NUM;i++){
#pragma xmp post (p(1), 1)
#pragma xmp post (p(2), 2)
#pragma xmp post (p(3), 3)
#pragma xmp post (p(4), 4)
    }
  }
  else if(id == 4){
    for(int i=0;i<NUM;i++){
#pragma xmp post (p(1), 12)
#pragma xmp post (p(2), 11)
#pragma xmp post (p(3), 6)
#pragma xmp post (p(4), 5)
    }
  }

  xmp_sync_all(&status);

  for(int i=0;i<NUM*4;i++){
#pragma xmp wait
  }
  xmp_sync_all(&status);

#pragma xmp task on p(1)
  {
    printf("PASS\n");
  }
  return 0;
}
