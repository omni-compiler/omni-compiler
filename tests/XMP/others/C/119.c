#include <stdlib.h>
#include <stdio.h>

int main(){
  int *a = malloc(100*sizeof(int));
  for(int i=0;i<100;i++)
    a[i] = i;

  int error = 1, *b;
  for(int i=0;i<100;i++){
    b = &a[i];
    if(i != *b)
      error = 0;
  }

  if(error)
    printf("PASS\n");
  else
    printf("ERROR\n");
  
  return 0;
}
