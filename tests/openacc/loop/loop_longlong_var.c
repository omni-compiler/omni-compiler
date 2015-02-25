#include<stdio.h>
#include<stdlib.h>

int main()
{
  const long long start = (long long)1024*1024*1024*6;
  const long long length = (long long)1024*1024;
  long long i;
  long long *a;
  long long *alloc;

  alloc = (long long *)malloc(sizeof(long long)*length);
  a = alloc - start;

#pragma acc data create(a[start:length])
  {
#pragma acc parallel loop
	for(i=start;i<start+length; i++){
	  a[i] = i;
	}
  
#pragma acc update host(a[start:length])
  }

  for(i=start; i<start+length; i++){
    if(a[i] != i){
      return 1;
    }
  }

  free(alloc);

  return 0;
}
