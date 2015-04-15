#include <stdio.h>
#include <stdlib.h>
int a:[*] = 2, result = 0;


int hoge(int a){
  return (a == -1);
}

int main(void)
{
  int a = -1;
  if(!hoge(a) || !(a == -1))
    result = -1;
  
  if(result == 0)
    printf("PASS\n");
  else{
    printf("ERROR\n");
    exit(1);
  }

  return 0;
}

