int main()
{
  double array[10];
  int i;
  int cond;

  for(i=0;i<10;i++){
    array[i] = (double)i;
  }

#pragma acc parallel copyin(array) if(123)
#pragma acc loop
  for(i=0;i<10;i++){
    array[i] = 0.0;
  }
  
  //verify
  for(i=0;i<10;i++){
    if(array[i] != (double)i){
      return 1;
    }
  }


#pragma acc parallel copyin(array) if(0)
#pragma acc loop
  for(i=0;i<10;i++){
    array[i] = 1.0;
  }

  //verify
  for(i=0;i<10;i++){
    if(array[i] != 1.0){
      return 2;
    }
  }

  cond = 456;

#pragma acc parallel copyin(array) if(cond)
#pragma acc loop
  for(i=0;i<10;i++){
    array[i] += 2.0;
  }

  //verify
  for(i=0;i<10;i++){
    if(array[i] != 1.0){
      return 3;
    }
  }

  cond = 0;

#pragma acc parallel copyin(array) if(cond)
#pragma acc loop
  for(i=0;i<10;i++){
    array[i] += 3.0;
  }

  //verify
  for(i=0;i<10;i++){
    if(array[i] != 4.0){
      return 4;
    }
  }

  return 0;
}
