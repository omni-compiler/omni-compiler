#include <stdio.h>

size_t getN(int n)
{
  unsigned char buf[n];
  return sizeof(buf);
}

int main()
{
  size_t sz[3];
  sz[0] = getN(1);    //sz == 1
  sz[1] = getN(256);  //sz == 256
  sz[2] = getN(1024); //sz == 1024

  if(sz[0] == 1 && sz[1] == 256 && sz[2] == 1024){
    printf("PASS\n");
  }
  else{
    printf("ERROR\n");
    return 1;
  }
  
  return 0;
}
