#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>
#define SIZE (64*1024)
#pragma xmp nodes p(*)
#pragma xmp template t(0:SIZE-1)
#pragma xmp distribute t(block) onto p
double a[SIZE];
#pragma xmp align a[i] with t(i)
#pragma xmp shadow a[1:1]
#define DUMMY_VAL -999
int result = 0;
int main()
{
  int num_comm = xmp_num_nodes();
  int node_id  = xmp_node_num();
  int width    = SIZE/num_comm;

#pragma xmp loop on t(i)
  for(int i=0; i<SIZE; i++)
    a[i] = node_id;

  int lo_shadow_index = width * (node_id-1) - 1;
  int hi_shadow_index = width * (node_id);
  a[lo_shadow_index] = a[hi_shadow_index] = DUMMY_VAL;
#pragma xmp barrier
#pragma acc data copy(a)
  {
#pragma xmp reflect_init (a) acc
#pragma xmp reflect_do (a) acc
    }
  
  // Check 
#pragma xmp loop on t(i)
  for(int i=0;i<SIZE;i++)
    if(a[i] != node_id)
      result = 1;

  if(node_id != 1)
    if(a[lo_shadow_index] != node_id - 1)
      result = 1;

  if(node_id != num_comm)
    if(a[hi_shadow_index] != node_id + 1)
      result = 1;

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

