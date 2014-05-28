#include<xmp.h>
#include<stdio.h> 
#include<stdlib.h>  
#pragma xmp nodes p(*)
int procs, id, mask1, val1, mask2, val2, i, w, l1, result = 0;

int main(void)
{
  if(xmp_num_nodes() > 31){
    printf("%s\n","You have to run this program by less than 32 nodes.");
    exit(1);
  }
  procs = xmp_num_nodes();
  id = xmp_node_num()-1;
  w=1;
  for(i=0;i<procs;i++){
    w=w*2;
  }
  for(i=0;i<w;i++){
    mask1 = 1 << id;
    val1 = i & mask1;
    if(val1 == 0){
      l1 = 0;
    }else{
      l1 = !0;
    }
#pragma xmp reduction(&&:l1)   
    if(i==w-1){
      if(!l1) result = -1; // NG
    }else{
      if(l1) result = -1; // NG
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
