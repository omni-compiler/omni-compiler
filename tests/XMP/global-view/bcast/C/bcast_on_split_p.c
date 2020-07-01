#include <xmp.h>
#include <stdio.h>      
#include <stdlib.h>
#pragma xmp nodes p[*]
static const int N=1000;
int i,j,aa[1000],a;
double bb[1000],b;
float cc[1000],c;
int procs,id,result=0;

int main(void)
{
  id = xmp_node_num();
  procs = xmp_num_nodes();
 
  j = 2;
  a = xmp_node_num();
  b = (double)a;
  c = (float)a;
  for(i=0;i<N;i++){
    aa[i] = a+i;
    bb[i] = (double)(a+i);
    cc[i] = (float)(a+i);
  }

#pragma xmp bcast (a) on p[1:procs-2]
#pragma xmp bcast (b) on p[1:procs-2]
#pragma xmp bcast (c) on p[1:procs-2]
#pragma xmp bcast (aa) on p[1:procs-2]
#pragma xmp bcast (bb) on p[1:procs-2]
#pragma xmp bcast (cc) on p[1:procs-2]

  if((id >= 2)&&(id<=procs-1)){
    if(a != j)         result = -1;
    if(b != (double)j) result = -1;
    if(c != (float)j)  result = -1;
    for(i=0;i<N;i++){
      if(aa[i] != j+i)           result = -1;
      if(bb[i] != (double)(j+i)) result = -1;
      if(cc[i] != (float)(j+i))  result = -1;
    }
  }
  else{
    if(a != xmp_node_num()) result = -1;
    if(b != (double)a)      result = -1;
    if(c != (float)a)       result = -1;
    for(i=0;i<N;i++){
      if(aa[i] != a+i)           result = -1;
      if(bb[i] != (double)(a+i)) result = -1;
      if(cc[i] != (float)(a+i))  result = -1;
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
