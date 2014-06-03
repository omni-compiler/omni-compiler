#include <xmp.h>
#include <stdio.h>
#include <stdlib.h>
#pragma xmp nodes p(*)
static const int N=1000;
int i,j,aa[1000],a;
double bb[1000],b;
float cc[1000],c;
int procs, result=0;

int main(void)
{
  procs = xmp_num_nodes();
  a = xmp_node_num();
  b = (double)a;
  c = (float)a;
  for(i=0;i<N;i++){
    aa[i] = a+i;
    bb[i] = (double)(a+i);
    cc[i] = (float)(a+i);
  }

#pragma xmp bcast (a)
#pragma xmp bcast (b) 
#pragma xmp bcast (c) 
#pragma xmp bcast (aa) 
#pragma xmp bcast (bb) 
#pragma xmp bcast (cc)
  j = 1;
  if(a != j)         result = -1;
  if(b != (double)j) result = -1;
  if(c != (float)j)  result = -1;
  for(i=0;i<N;i++){
    if(aa[i] != j+i)           result = -1;
    if(bb[i] != (double)(j+i)) result = -1;
    if(cc[i] != (float)(j+i))  result = -1;
   }

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
