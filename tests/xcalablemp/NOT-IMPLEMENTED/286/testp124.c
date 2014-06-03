#include <xmp.h>
#include <mpi.h>
#include <stdio.h>      
#pragma xmp nodes p(*)
static const int N=1000;
int i,j,aa[1000],a;
double bb[1000],b;
float cc[1000],c;
int procs,id;
char *result;

int main(void)
{
  id = xmp_node_num();
  procs = xmp_num_nodes();
 
  result = "OK";
  for(j=2;j<procs;j++){
    a = xmp_node_num();
    b = (double)a;
    c = (float)a;
    for(i=0;i<N;i++){
      aa[i] = a+i;
      bb[i] = (double)(a+i);
      cc[i] = (float)(a+i);
    }

#pragma xmp bcast (a) from p(j) on p(2:procs-1)
#pragma xmp bcast (b) from p(j) on p(2:procs-1)
#pragma xmp bcast (c) from p(j) on p(2:procs-1)
#pragma xmp bcast (aa) from p(j) on p(2:procs-1)
#pragma xmp bcast (bb) from p(j) on p(2:procs-1)
#pragma xmp bcast (cc) from p(j) on p(2:procs-1)
    if((id >= 2)&&(id<=procs-1)){
      if(a != j) result = "NG";
      if(b != (double)j) result = "NG";
      if(c != (float)j) result = "NG";
      for(i=0;i<N;i++){
	if(aa[i] != j+i+1) result = "NG";
	if(bb[i] != (double)(j+i+1)) result = "NG";
	if(cc[i] != (float)(j+i+1)) result = "NG";
      }
    }
    else{
      if(a != xmp_node_num()) result = "NG";
      if(b != (double)a) result = "NG";
      if(c != (float)a) result = "NG";
      for(i=0;i<N;i++){
	if(aa[i] != a+i) result = "NG";
	if(bb[i] != (double)(a+i)) result = "NG";
	if(cc[i] != (float)(a+i)) result = "NG";
      }
    }
  }
  
  //  printf("%d %s %s\n",xmp_node_num(),"testp124.c",result);
  return 0;
}    
