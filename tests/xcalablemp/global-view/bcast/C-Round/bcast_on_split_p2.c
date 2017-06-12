#include <xmp.h>
#include <stdlib.h>
#include <stdio.h>      
#pragma xmp nodes p(4,*)
static const int N=1000;
int i,j,k,aa[1000],a;
double bb[1000],b;
float cc[1000],c;
int procs,id,procs2,ans,result=0;

int main(void)
{
  id = (xmp_node_num()-1)%4+1;
  procs = xmp_num_nodes();
  procs2 = procs/4;
 
  a = xmp_node_num();
  b = (double)a;
  c = (float)a;
  for(i=0;i<N;i++){
    aa[i] = a+i;
    bb[i] = (double)(a+i);
    cc[i] = (float)(a+i);
  }
#pragma xmp bcast (a) on p(2:3,1:procs2)
#pragma xmp bcast (b) on p(2:3,1:procs2)
#pragma xmp bcast (c) on p(2:3,1:procs2)
#pragma xmp bcast (aa) on p(2:3,1:procs2)
#pragma xmp bcast (bb) on p(2:3,1:procs2)
#pragma xmp bcast (cc) on p(2:3,1:procs2)
  ans = 2;
  if((id >= 2)&&(id <= 3)){
    if(a != ans) result = -1;
    if(b != (double)ans) result = -1;
    if(c != (float)ans) result = -1;
    for(i=0;i<N;i++){
      if(aa[i] != ans+i) result = -1;
      if(bb[i] != (double)(ans+i)) result = -1;
      if(cc[i] != (float)(ans+i)) result = -1;
    }
  }else{
    if(a != xmp_node_num()) result = -1;
    if(b != (double)a) result = -1;
    if(c != (float)a) result = -1;
    for(i=0;i<N;i++){
      if(aa[i] != a+i) result = -1;
      if(bb[i] != (double)(a+i)) result = -1;
      if(cc[i] != (float)(a+i)) result = -1;
    }
  }

#pragma xmp reduction(+:result)
#pragma xmp task on p(1,1)
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
