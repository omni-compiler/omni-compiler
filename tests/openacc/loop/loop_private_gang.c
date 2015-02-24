int main()
{
  int a = 1;
  int array[100];
  int i,j;
  
  for(i=0;i<100;i++){
    array[i] = i;
  }

#pragma acc parallel
  {
#pragma acc loop gang private(a)
    for(i=0;i<10;i++){
      a = i;
#pragma acc loop vector
      for(j=0;j<10;j++){
	array[i*10 + j] = j + a;
      }
    }
  }

  //verify
  for(j=0;j<10;j++){
    if(array[20+j] != 2+j){
      return 1;
    }
  }
  
  return 0;
}
