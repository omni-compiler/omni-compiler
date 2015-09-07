#include <stdio.h>
#include <xmp.h>
#define MAX_TAG 4
int status;
#pragma xmp nodes p(4)

void init_tag(int tag[MAX_TAG]){
  int i;
  for(i=0;i<MAX_TAG;i++)    tag[i] = i%15;  // On the K conputer, 0 <= tag <= 14
}

void post_wait_local(){
#pragma xmp task on p(1)
  {
#pragma xmp post (p(1), 9)
#pragma xmp post (p(1), 8)
#pragma xmp post (p(1), 0)

#pragma xmp wait (p(1), 8)
#pragma xmp wait (p(1))
#pragma xmp wait
  }
  xmp_sync_all(&status);
}

void post_wait_p2p(int tag[MAX_TAG]){
  init_tag(tag);
#pragma xmp task on p(1)
  {
#pragma xmp post (p(2), tag[3])
#pragma xmp post (p(2), tag[1])
#pragma xmp post (p(2), tag[2])
  }

  xmp_sync_all(&status);

#pragma xmp task on p(2)
  {
#pragma xmp wait (p(1), tag[1])
#pragma xmp wait (p(1), tag[3])
#pragma xmp wait (p(1), tag[2])
  }
  xmp_sync_all(&status);
}

void post_wait_nodes(){
  int tag2 = 7;
#pragma xmp task on p(1)
{
  int target_node = 3;
#pragma xmp post (p(target_node), tag2)
 }

#pragma xmp task on p(2)
 {
   int target_node[100][3];
   target_node[3][2] = 3;
   int k = target_node[3][2];
#pragma xmp post (p(k), tag2)
 }

#pragma xmp task on p(3)
 {
#pragma xmp wait (p(1), tag2)
#pragma xmp wait (p(2), tag2)
 }

 xmp_sync_all(&status);
}

int main(){
  post_wait_local();

  int tag[MAX_TAG];
  post_wait_p2p(tag);

  post_wait_nodes();

#pragma xmp task on p(1)
  {
    printf("PASS\n");
  }
  return 0;
}
