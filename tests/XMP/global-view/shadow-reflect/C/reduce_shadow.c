#include <stdio.h>
#include <stdlib.h>
#include "xmp.h"

int main(){
#pragma xmp nodes p[4][4]
#pragma xmp template t[:][:]
#pragma xmp distribute t[gblock(*)][gblock(*)] onto p
  int m[4] = {4, 4, 8, 16};
  int n = 32;
  int (*a)[n], (*b)[n];
#pragma xmp align a[i][j] with t[i][j]
#pragma xmp shadow a[1][1]
#pragma xmp template_fix[gblock(m)][gblock(m)] t[0:n][0:n]
  a = (int (*)[n])xmp_malloc(xmp_desc_of(a), n, n);
  b = (int (*)[n])malloc(sizeof(int)*n*n);
  
#pragma xmp array on t[0:n][0:n]
  a[0:n][0:n] = 1;

  for (int i = 0; i < n; i++){
    for (int j = 0; j < n; j++){
      b[i][j] = 1;
      if (i == 0 || i == 3 || 
	  i == 4 || i == 7 || 
	  i == 8 || i == 15 || 
	  i == 16 || i == 31){
	b[i][j]++;
      }
      if (j == 0 || j == 3 || 
	  j == 4 || j == 7 || 
	  j == 8 || j == 15 || 
	  j == 16 || j == 31){
	b[i][j]++;
      }
    }
  }

#pragma xmp reflect (a) width(/periodic/1, /periodic/1)
#pragma xmp reduce_shadow (a) width(/periodic/1,/periodic/1) async(10)
#pragma xmp wait_async (10)
  int result = 0;
  
#pragma xmp loop (i,j) on t[i][j] reduction(+:result)
  for (int i = 0; i < n; i++){
    for (int j = 0; j < n; j++){
      if (a[i][j] != b[i][j]) result = 1;
    }
  }

#pragma xmp task on p[1][1]
  {
    if (result == 0){
      printf("PASS\n");
    }
    else{
      fprintf(stderr, "ERROR\n");
      exit(1);
    }
  }

  return 0;
  
}
