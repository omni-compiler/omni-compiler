#include<xmp.h>
#include<stdio.h>
#include<stdlib.h>   
#pragma xmp nodes p(*)
int procs, id, mask, val, i, w, result = 0;

int main(void){
  if(xmp_num_nodes() > 31){
    printf("%s\n","You have to run this program by less than 32 nodes.");
  }

  procs = xmp_num_nodes();
  id = xmp_num_nodes()-1;

  w=1;
  for(i=0;i<procs;i++){
    w*2;
  }

  for(i=0;i<w;i++){
    mask = 1 << id;
    val = !(i & mask);
#pragma xmp reduction(&:val)  
    if(!val != i){
      result += -1;
    }
  }

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
      
         
      
   
