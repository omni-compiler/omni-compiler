#include <xmp.h>
#include <stdio.h>  
#include <stdlib.h> 
static const int N=1000;
#pragma xmp nodes p(*)
#pragma xmp template t1(0:N-1,0:N-1,0:N-1)
#pragma xmp template t2(0:N-1,0:N-1,0:N-1)
#pragma xmp template t3(0:N-1,0:N-1,0:N-1)
#pragma xmp distribute t1(cyclic,*,*) onto p
#pragma xmp distribute t2(*,cyclic,*) onto p
#pragma xmp distribute t3(*,*,cyclic) onto p
int a[N],sa=0;
double b[N],sb=0.0;
float c[N],sc=0.0;
int i,*w,ans=0,procs,result=0;
#pragma xmp align a[i] with t1(i,*,*)
#pragma xmp align b[i] with t2(*,i,*)
#pragma xmp align c[i] with t3(*,*,i)

int main(void)
{
#pragma xmp loop (i) on t1(i,:,:)
  for(i=0;i<N;i++)
    a[i]=xmp_node_num();

#pragma xmp loop (i) on t2(:,i,:)
  for(i=0;i<N;i++)
    b[i]=(double)xmp_node_num();

#pragma xmp loop (i) on t3(:,:,i)
  for(i=0;i<N;i++)
    c[i]=(float)xmp_node_num();

#pragma xmp loop on t1(i,*,*) reduction(+:sa)
  for(i=0;i<N;i++)
    sa+=a[i];

#pragma xmp loop on t2(*,i,*) reduction(+:sb)
  for(i=0;i<N;i++)
    sb+=b[i];

#pragma xmp loop on t3(*,*,i) reduction(+:sc)
  for(i=0;i<N;i++)
    sc+=c[i];
  
  procs = xmp_num_nodes();
  w = (int *)malloc(procs*sizeof(int));
  if(N%procs == 0){
    for(i=1;i<procs+1;i++){
      w[i] = N/procs;
    }
  }else{
    for(i=1;i<procs+1;i++){
      if(i <= N%procs){
	w[i]=N/procs+1;
      }else{
	w[i]=N/procs;
      }
    }
  }

  for(i=1;i<procs+1;i++)
    ans += i*w[i];

  if( (sa != ans) || abs(sb-(double)ans) >= 0.0000001 || abs(sc-(float)ans) >= 0.0001 )
    result = -1;

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
