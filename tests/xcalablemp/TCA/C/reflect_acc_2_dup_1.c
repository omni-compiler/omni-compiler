#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>
#define N 64
double a[N][N];
#pragma xmp nodes p(*)
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(block) onto p
#pragma xmp align a[i][*] with t(i)
#pragma xmp shadow a[1:1][0]
#define DUMMY_VAL -999
#define DUMMY_NODE_ID -999
int result = 0;

int main(int argc, char **argv)
{
  int comm_size = xmp_num_nodes();
  int node_id   = xmp_node_num();

#pragma xmp loop on t(i)
  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      a[i][j] = node_id;

  int lo_index = N/comm_size*(node_id-1);
  int hi_index = N/comm_size*node_id-1;
  int lo_shadow_index = (node_id != 1)?         lo_index-1 : DUMMY_NODE_ID;
  int hi_shadow_index = (node_id != comm_size)? hi_index+1 : DUMMY_NODE_ID;

  if(lo_shadow_index != DUMMY_NODE_ID)
    for(int j=0;j<N;j++)
      a[lo_shadow_index][j] = node_id;

  if(hi_shadow_index != DUMMY_NODE_ID)
    for(int j=0;j<N;j++)
      a[hi_shadow_index][j] = node_id;

#pragma xmp barrier
#pragma acc data copy(a)
  {
#pragma xmp reflect_init (a) acc
#pragma xmp reflect_do (a) acc
  }

#pragma xmp loop on t(i)
  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      if(a[i][j] != node_id)
	result = 1;

  int lo_node_id = (node_id != 1)?         node_id-1 : DUMMY_NODE_ID;
  int hi_node_id = (node_id != comm_size)? node_id+1 : DUMMY_NODE_ID;

  if(lo_node_id != DUMMY_NODE_ID)
    for(int j=0;j<N;j++)
      if(a[lo_shadow_index][j] != lo_node_id)
	result = 1;

  if(hi_node_id != DUMMY_NODE_ID)
    for(int j=0;j<N;j++)
      if(a[hi_shadow_index][j] != hi_node_id)
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
