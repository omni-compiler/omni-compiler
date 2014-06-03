/*testp139.c*/
/*bcast指示文のテスト:fromはnode-ref,onはnode-refかつ部分ノード*/
#include<xmp.h>
#include<stdio.h>      
#pragma xmp nodes p(4,*)
static const int N=1000;
int i,j,k,aa[1000],a;
double bb[1000],b;
float cc[1000],c;
int procs,id,procs2,ans;
char *result;

int main(void)
{
  id = xmp_node_num();
  procs = xmp_num_nodes();
  procs2 = procs/4;
  result = "OK";
  for(k=1;k<procs2+1;k++){
    for(j=2;j<4;j++){
      a = xmp_node_num();
      b = (double)a;
      c = (float)a;
      for(i=0;i<N;i++){
	aa[i] = a+i-1;
	bb[i] = (double)(a+i-1);
	cc[i] = (float)(a+i-1);
      }

#pragma xmp bcast (a) from p(j,k) on p(2:3,1:procs2)
      //#pragma xmp bcast (b) from p(j,k) on p(2:3,1:procs2)
      //#pragma xmp bcast (c) from p(j,k) on p(2:3,1:procs2)
      //#pragma xmp bcast (aa) from p(j,k) on p(2:3,1:procs2)
      //#pragma xmp bcast (bb) from p(j,k) on p(2:3,1:procs2)
      //#pragma xmp bcast (cc) from p(j,k) on p(2:3,1:procs2)
      ans = (k-1)*4+j;
      if((id >= 2)&&(id <= procs-1)){
	if(a != ans) result = "NG";
	if(b != (double)ans) result = "NG";
	if(c != (float)ans) result = "NG";
	for(i=0;i<N;i++){
	  if(aa[i] != ans+i-1) result = "NG";
	  if(bb[i] != (double)(ans+i-1)) result = "NG";
	  if(cc[i] != (float)(ans+i-1)) result = "NG";
	}
      }else{
	if(a != xmp_node_num()) result = "NG";
	if(b != (double)a) result = "NG";
	if(c != (float)a) result = "NG";
	for(i=0;i<N;i++){
	  if(aa[i] != a+i-1) result = "NG";
	  if(bb[i] != (double)(a+i-1)) result = "NG";
	  if(cc[i] != (float)(a+i-1)) result = "NG";
	}
      }
    }
  }
  printf("%d %s %s\n",xmp_node_num(),"testp139.c",result);
  return 0;
}    
         
      
   
