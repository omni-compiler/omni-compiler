#include <stdio.h>
#include <string.h>
int main(){
  if(strcmp("P A S S",PASS) == 0){
    printf("PASS\n");
    return 0;
  }
  else{
    printf("ERROR\n");
    return 1;
  }
}
