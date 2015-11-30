#include <stdio.h>
#include <xmp.h>
#define NUM_NODES 4
#define NUM_ITERS 8

#pragma xmp nodes p(NUM_NODES)

int coarray[1]:[*];

int main()
{
  int image_num = xmp_node_num();
  int i, iter;

  if(xmp_num_nodes() < 2){
    printf("num nodes is too few\n");
    return 1;
  }


  int images[NUM_NODES-1];
  int idx = 0;
  for(i=1;i<=NUM_NODES;i++){
    if(i == image_num) continue;
    images[idx++] = i;
  }

  coarray[0] = image_num * 2 + 1;

  for(iter = 0; iter < NUM_ITERS; iter++){
    xmp_sync_images(NUM_NODES-1, images, NULL);

    int sum = 0;
    for(i = 1; i <= NUM_NODES; i++){
      if(i == image_num) continue;
      int tmp;
      tmp = coarray[0]:[i];
      sum += tmp;
    }

    xmp_sync_images(NUM_NODES-1, images, NULL);
    coarray[0] = sum;
  }

  //verify
  if(coarray[0] != 39361 + 2 * image_num){
    printf("image=%d, invalid result=%d\n", image_num,  coarray[0]);
    return 1;
  }

  if(image_num == 1){
    printf("PASS\n");
  }

  return 0;
}
