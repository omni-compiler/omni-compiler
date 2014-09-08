#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>
#define N 64
double a[N][N];
#define P 2
#define Q 4
#pragma xmp nodes p(P,Q)
#pragma xmp template t(0:N-1,0:N-1)
#pragma xmp distribute t(block,block) onto p
#pragma xmp align a[i][j] with t(j, i)
#pragma xmp shadow a[1:1][1:1]
#define DUMMY_VAL -999
#define DUMMY_NODE_ID -999
int result = 0;
int main(int argc, char **argv)
{
  int comm_size = xmp_num_nodes();
  int node_id   = xmp_node_num();

#pragma xmp loop (j, i) on t(j, i)
  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      a[i][j] = xmp_node_num();

  int node_id_1 = (node_id-1)/P;
  int node_id_2 = (node_id-1)%P;
  int lo_shadow_index_1 = (N/Q) * node_id_1 - 1;
  int hi_shadow_index_1 = (N/Q) * (node_id_1+1);
  int lo_shadow_index_2 = (N/P) * node_id_2 - 1;
  int hi_shadow_index_2 = (N/P) * (node_id_2+1);

#pragma xmp loop (j, i) on t(j, i)
  for(int i=lo_shadow_index_1;i<lo_shadow_index_1+1;i++)
    for(int j=0;j<N;j++)
      a[i][j] = DUMMY_VAL;

#pragma xmp loop (j, i) on t(j, i)
  for(int i=hi_shadow_index_1;i<hi_shadow_index_1+1;i++)
    for(int j=0;j<N;j++)
      a[i][j] = DUMMY_VAL;

#pragma xmp loop (j, i) on t(j, i)
  for(int i=0;i<N;i++)
    for(int j=lo_shadow_index_2;j<lo_shadow_index_2+1;j++)
      a[i][j] = DUMMY_VAL;

#pragma xmp loop (j, i) on t(j, i)
  for(int i=0;i<N;i++)
    for(int j=hi_shadow_index_2;j<hi_shadow_index_2+1;j++)
      a[i][j] = DUMMY_VAL;

#pragma xmp barrier
#pragma acc data copy(a)
  {
#pragma xmp reflect_init (a) acc
#pragma xmp reflect_do (a) acc
  }

#pragma xmp loop (j, i) on t(j, i)
  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      if(a[i][j] != node_id)
	result = 1;

  int hi_node_id_1 = (node_id <= comm_size-P)? node_id+P : DUMMY_NODE_ID;
  int lo_node_id_1 = (node_id > P)?            node_id-P : DUMMY_NODE_ID;
  int hi_node_id_2 = ((node_id-1)%P != 1)?     node_id+1 : DUMMY_NODE_ID;
  int lo_node_id_2 = ((node_id-1)%P != 0)?     node_id-1 : DUMMY_NODE_ID;

  if(lo_node_id_1 != DUMMY_NODE_ID)
    {
#pragma xmp loop (j, i) on t(j, i)
      for(int i=lo_shadow_index_1;i<lo_shadow_index_1+1;i++)
	for(int j=0;j<N;j++)
	  if(a[i][j] != lo_node_id_1){
	    result = 1;
	    printf("[%d]i=%d j=%d\n",node_id,i,j);}
    }

  if(hi_node_id_1 != DUMMY_NODE_ID)
    {
#pragma xmp loop (j, i) on t(j, i)
      for(int i=hi_shadow_index_1;i<hi_shadow_index_1+1;i++)
	for(int j=0;j<N;j++)
	  if(a[i][j] != hi_node_id_1)
	    result = 1;
    }

  if(lo_node_id_2 != DUMMY_NODE_ID)
    {
#pragma xmp loop (j, i) on t(j, i)
      for(int i=0;i<N;i++)
	for(int j=lo_shadow_index_2;j<lo_shadow_index_2+1;j++)
	  if(a[i][j] != lo_node_id_2)
	    result = 1;
    }

  if(hi_node_id_2 != DUMMY_NODE_ID)
    {
#pragma xmp loop (j, i) on t(j, i)
      for(int i=0;i<N;i++)
	for(int j=hi_shadow_index_2;j<hi_shadow_index_2+1;j++)
	  if(a[i][j] != hi_node_id_2)
	    result = 1;
    }

#pragma xmp reduction(+:result)
#pragma xmp task on p(1,1)
  {
    if(result == 0){
      printf("PASS\n");
    }
    else{
      fprintf(stderr, "ERROR %d\n", result);
      exit(1);
    }
  }

  return 0;
}
