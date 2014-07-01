#include <xmp.h>
#include <stdlib.h>
#include <stdio.h> 
static const int N=1000;
#pragma xmp nodes p(*)
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(block) onto p
int i,j,aa[1000],a;
double bb[1000],b;
float cc[1000],c;
int procs,ans,w, result=0;

int main(void)
{
  procs = xmp_num_nodes();
  if(N%procs == 0){
    w = N/procs;
  }else{
    w = N/procs+1;
  }

  for(j=0;j<N;j++){
    a = xmp_node_num();
    b = (double)a;
    c = (float)a;
    for(i=0;i<N;i++){
      aa[i] = a+i-1;
      bb[i] = (double)(a+i-1);
      cc[i] = (float)(a+i-1);
    }

#pragma xmp bcast (a) from t(j) 
#pragma xmp bcast (b) from t(j)
#pragma xmp bcast (c) from t(j)
#pragma xmp bcast (aa) from t(j)
#pragma xmp bcast (bb) from t(j)
#pragma xmp bcast (cc) from t(j)

    ans = j/w+1;
    if(a != ans) result = -1;
    if(b != (double)ans) result = -1;
    if(c != (float)ans) result = -1;
    for(i=0;i<N;i++){
      if(aa[i] != ans+i-1) 
	result = -1;
      if(bb[i] != (double)(ans+i-1))
	result = -1;
      if(cc[i] != (float)(ans+i-1))
	result = -1;
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

