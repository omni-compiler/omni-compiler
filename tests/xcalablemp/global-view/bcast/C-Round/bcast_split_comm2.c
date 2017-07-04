#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>
int row_num = 4;
int col_num = 2;
#pragma xmp nodes p(row_num,col_num)
#pragma xmp nodes col(*) = p(:,*)
int id, result=0;

int main(){
  int i, a;

  for(i=1;i<row_num+1;i++){
    id = a = xmp_node_num();
#pragma xmp bcast (a) from col(i) on col(:)
    if(((id-1)/row_num)*row_num+(i-1)%row_num+1 != a)
      result = -1;
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
