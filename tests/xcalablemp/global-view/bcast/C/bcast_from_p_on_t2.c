#include <xmp.h>
#include <stdlib.h>
#include <stdio.h>      
static const int N=1000;
#pragma xmp nodes p(*)
#pragma xmp template t(1000,1000)
#pragma xmp distribute t(*,cyclic) onto p
int i,j,aa[1000],a;
double bb[1000],b;
float cc[1000],c;
int procs,result=0;

int main(void)
{
  procs = xmp_num_nodes();
  for(j=1;j<procs+1;j++){
    a = xmp_node_num();
    b = (double)a;
    c = (float)a;
    for(i=0;i<N;i++){
      aa[i] = a+i;
      bb[i] = (double)(a+i);
      cc[i] = (float)(a+i);
    }
#pragma xmp bcast (a) from p(j) on t(:,:)
#pragma xmp bcast (b) from p(j) on t(:,:)
#pragma xmp bcast (c) from p(j) on t(:,:)
#pragma xmp bcast (aa) from p(j) on t(:,:)
#pragma xmp bcast (bb) from p(j) on t(:,:)
#pragma xmp bcast (cc) from p(j) on t(:,:)
    if(a != j) result = -1;
    if(b != (double)j) result = -1;
    if(c != (float)j) result = -1;
    for(i=0;i<N;i++){
      aa[i] = a+i;
      bb[i] = (double)(a+i);
      cc[i] = (float)(a+i);
    }
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
