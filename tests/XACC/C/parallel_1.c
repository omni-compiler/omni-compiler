#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>
#define SIZE (64*1024)
#pragma xmp nodes p(*)
#pragma xmp template t(0:SIZE-1)
#pragma xmp distribute t(block) onto p
double a[SIZE];

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

#pragma xmp barrier

#pragma xmp loop on t(i)
#pragma acc parallel loop copy(a)
    for(int i=0; i<SIZE; i++)
      a[i] += i;
  
  // Check 
#pragma xmp loop on t(i)
  for(int i=0;i<SIZE;i++)
    if(a[i] != i + node_id)
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

