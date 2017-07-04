#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>
#define SIZE (64*1024)
#define LO_SHADOW_WIDTH 4
#define HI_SHADOW_WIDTH 3
#define LO_REFLECT_WIDTH 2
#define HI_REFLECT_WIDTH 1
#pragma xmp nodes p[*]
#pragma xmp template t[SIZE]
#pragma xmp distribute t[block] onto p
double a[SIZE];
#pragma xmp align a[i] with t[i]
#pragma xmp shadow a[LO_SHADOW_WIDTH:HI_SHADOW_WIDTH]
#define DUMMY_VAL -999
int result = 0;
int main()
{
  int num_comm = xmp_num_nodes();
  int node_id  = xmp_node_num();
  int width    = SIZE/num_comm;

#pragma xmp loop on t[i]
  for(int i=0; i<SIZE; i++)
    a[i] = node_id;

  int lo_shadow_index = width * (node_id-1) - LO_SHADOW_WIDTH;
  int hi_shadow_index = width * (node_id);
  for(int i=0; i<LO_SHADOW_WIDTH; i++){
    a[lo_shadow_index + i] = DUMMY_VAL;
  }
  for(int i=0; i<HI_SHADOW_WIDTH; i++){
    a[hi_shadow_index + i] = DUMMY_VAL;
  }

#pragma xmp barrier
#pragma acc data copy(a)
  {
#pragma xmp reflect (a) width(LO_REFLECT_WIDTH:HI_REFLECT_WIDTH) acc
  }

  // Check
#pragma xmp loop on t[i]
  for(int i=0;i<SIZE;i++)
    if(a[i] != node_id)
      result = 1;

  if(node_id != 1){
    for(int i=0; i<LO_SHADOW_WIDTH; i++){
      //      printf("%d: lo[%d]=%f\n", node_id, i, a[lo_shadow_index+i]);
      if(i < LO_SHADOW_WIDTH-LO_REFLECT_WIDTH){
	if(a[lo_shadow_index+i] != DUMMY_VAL) result = 2;
      }else{
	if(a[lo_shadow_index+i] != node_id - 1) result = 3;
      }
    }
  }
  if(node_id != num_comm){
    for(int i=0; i<HI_SHADOW_WIDTH; i++){
      //      printf("%d: hi[%d]=%f\n", node_id, i, a[hi_shadow_index+i]);
      if(i >= HI_REFLECT_WIDTH){
	if(a[hi_shadow_index+i] != DUMMY_VAL) result = 4;
      }else{
	if(a[hi_shadow_index+i] != node_id + 1) result = 5;
      }
    }
  }

#pragma xmp reduction(+:result)
#pragma xmp task on p[0]
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

