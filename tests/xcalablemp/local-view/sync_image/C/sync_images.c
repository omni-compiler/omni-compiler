#include <stdio.h>
#include <xmp.h>
#define NUM_NODES 4
#define NUM_ITERS 8
#pragma xmp nodes p[NUM_NODES]
int coarray[1]:[*];

int main()
{
  int image_num = xmpc_this_image();
  int idx = 0, images[NUM_NODES-1];
  
  for(int i=0;i<NUM_NODES;i++){
    if(i == image_num) continue;
    images[idx++] = i;
  }

  coarray[0] = (image_num+1) * 2 + 1;

  for(int iter=0;iter<NUM_ITERS;iter++){
    //    xmp_sync_images(NUM_NODES-1, images, NULL);
    xmp_sync_all(NULL);
    int sum = 0;
    for(int i=0;i<NUM_NODES;i++){
      if(i == image_num) continue;
      int tmp;
      tmp = coarray[0]:[i];
      sum += tmp;
    }
    xmp_sync_all(NULL);
    //    xmp_sync_images(NUM_NODES-1, images, NULL);
    coarray[0] = sum;
  }

  //verify
  if(coarray[0] != 39361 + 2 * (image_num+1)){
    printf("image=%d, invalid result=%d\n", image_num,  coarray[0]);
    return 1;
  }

  if(image_num == 1){
    printf("PASS\n");
  }

  return 0;
}
