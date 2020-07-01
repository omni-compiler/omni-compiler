#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>
int row_num = 4;
int col_num = 2;
#pragma xmp nodes p[col_num][row_num]
#pragma xmp nodes row[*] = p[:][*]
int id, result=0;

int main(){
  int i, a;

  for(i=1;i<col_num+1;i++){
    id = a = xmp_node_num();
#pragma xmp bcast (a) from row[i-1] on row[:]
    if((((id-1)%row_num)+1) + (i-1)*row_num != a)
      result = -1;
  }

#pragma xmp reduction(+:result)
#pragma xmp task on p[0][0]
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
