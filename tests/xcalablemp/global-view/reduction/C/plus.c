#include <xmp.h>
#include <stdio.h>
#include <stdlib.h>   
#include <string.h>
static const int N=1000;
#pragma xmp nodes p(*)
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(cyclic) onto p
int a[N],   sa=0, procs, w, remain, *w1, i, j, result = 0;
double b[N],sb=0.0;
float c[N], sc=0.0;
#pragma xmp align a[i] with t(i)
#pragma xmp align b[i] with t(i)
#pragma xmp align c[i] with t(i)

int main(void){
  if(xmp_num_nodes() < 4){
    fprintf(stderr, "%s\n","You have to run this program by more than 4 nodes.");
    exit(1);
  }

#pragma xmp loop on t(i)
  for(i=0;i<N;i++){
    a[i]=1;
    b[i]=0.5;
    c[i]=0.01;
  }

#pragma xmp loop on t(i)
  for(i=0;i<N;i++)
    sa = sa+a[i];
#pragma xmp reduction (+:sa) on p(1:2) 
  
#pragma xmp loop on t(i)
  for(i=0;i<N;i++)
    sb = sb+b[i];
#pragma xmp reduction(+:sb) on p(2:3) 

#pragma xmp loop on t(i)
  for(i=0;i<N;i++)
    sc = sc+c[i];
#pragma xmp reduction(+:sc) on p(3:4) 
  
  procs = xmp_num_nodes();
  w1 = (int *)malloc((procs+1)*sizeof(int));
  if((N%procs) == 0){
    for(i=1;i<procs+1;i++){
      w1[i] = N/procs;
    }
  }else{
    w = N%procs;
    for(i=1;i<procs+1;i++){
      if(i<=w){
	w1[i] = N/procs+1;
      }else{
	w1[i] = N/procs;
      }
    }
  }

  if(xmp_node_num() == 1){
    if((sa != (w1[1]+w1[2])) || (abs(sb-((double)w1[1]*0.5)) > 0.000001) || (abs(sc-((float)w1[1]*0.01)) > 0.000001)){
      result = -1; // NG
    }
  }else if(xmp_node_num() == 2){
    if((sa != (w1[1]+w1[2])) || (abs(sb-((double)(w1[2]+w1[3])*0.5)) > 0.000001) || (abs(sc-((float)w1[2]*0.01)) > 0.000001)){
      result = -1; // NG
    }
  }else if(xmp_node_num() == 3){
    if((sa != w1[3]) || (abs(sb-((double)(w1[2]+w1[3])*0.5)) > 0.000001) || (abs(sc-((float)(w1[3]+w1[4])*0.01)) > 0.000001)){
      result = -1; // NG
    }
  }else if(xmp_node_num() == 4){
    if((sa != w1[4]) || (abs(sb-((double)w1[4]*0.5)) > 0.000001) || (abs(sc-((float)(w1[3]+w1[4])*0.01)) > 0.000001)){
      result = -1; // NG
    }
  }else{
    i=xmp_node_num();
    if((sa != w1[i]) || (abs(sb-((double)w1[i]*0.5)) > 0.000001) || (abs(sc-((float)w1[i]*0.01)) > 0.000001)){
      result = -1; // NG
    }
  }
  free(w1);

#pragma xmp reduction(+:result)

#pragma xmp task on p(1)
  {
    if(result == 0){
      printf("PASS\n");
    } else{
      printf("ERROR\n");
      exit(1);
    }
  }

  return 0;
}
