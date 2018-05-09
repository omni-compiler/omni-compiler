#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>
#define NUM 1000
int status;
#pragma xmp nodes p[4]

int main()
{
  int id = xmpc_this_image();

  if(id == 0){
    for(int i=0;i<NUM;i++){
#pragma xmp post (p[0], 2)
#pragma xmp post (p[1], 9)
#pragma xmp post (p[2], 8)
#pragma xmp post (p[3], 0)
    }
  }
  else if(id == 1){
    for(int i=0;i<NUM;i++){
#pragma xmp post (p[0], 9)
#pragma xmp post (p[1], 2)
#pragma xmp post (p[2], 1)
#pragma xmp post (p[3], 4)
    }
  }
  else if(id == 2){
    for(int i=0;i<NUM;i++){
#pragma xmp post (p[0], 1)
#pragma xmp post (p[1], 2)
#pragma xmp post (p[2], 3)
#pragma xmp post (p[3], 4)
    }
  }
  else if(id == 3){
    for(int i=0;i<NUM;i++){
#pragma xmp post (p[0], 12)
#pragma xmp post (p[1], 11)
#pragma xmp post (p[2], 6)
#pragma xmp post (p[3], 5)
    }
  }

  xmp_sync_all(NULL);

  if(id == 0){
    for(int i=0;i<NUM;i++){
#pragma xmp wait (p[0], 2)
#pragma xmp wait (p[1], 9)
#pragma xmp wait (p[2], 1)
#pragma xmp wait (p[3], 12)
    }
  }
  else if(id == 1){
    for(int i=0;i<NUM;i++){
#pragma xmp wait (p[0], 9)
#pragma xmp wait (p[1], 2)
#pragma xmp wait (p[2], 2)
#pragma xmp wait (p[3], 11)
    }
  }
  else if(id == 2){
    for(int i=0;i<NUM;i++){
#pragma xmp wait (p[0], 8)
#pragma xmp wait (p[1], 1)
#pragma xmp wait (p[2], 3)
#pragma xmp wait (p[3], 6)
    }
  }
  else if(id == 3){
    for(int i=0;i<NUM;i++){
#pragma xmp wait (p[0], 0)
#pragma xmp wait (p[1], 4)
#pragma xmp wait (p[2], 4)
#pragma xmp wait (p[3], 5)
    }
  }

  xmp_sync_all(NULL);

#pragma xmp task on p[0]
  {
    printf("PASS\n");
  }
  return 0;
}
