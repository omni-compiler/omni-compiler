int main()
{
  double array[100];
  int i,j;
  
  for(i=0;i<100;i++){
    array[i] = (double)i;
  }

  double sum = 0;

#pragma acc parallel loop reduction(+:sum)
  for(i=0;i<100;i++){
    sum += array[i];
  }
  //verify
  if(sum != 4950.0) return 1;


  sum = 0;
#pragma acc parallel loop reduction(+:sum) copy(sum)
  for(i=0;i<100;i++){
    sum += array[i];
  }
  //verify
  if(sum != 4950.0) return 2;


  sum = 0;
#pragma acc data copy(sum)
  {
#pragma acc parallel loop reduction(+:sum)
    for(i=0;i<50;i++){
      sum += array[i];
    }
  }
  //verify
  if(sum != 1225.0) return 3;


  sum = 100;
#pragma acc data copy(sum)
  {
#pragma acc parallel loop reduction(+:sum)
    for(i=50;i<100;i++){
      sum += array[i];
    }
  }
  //vefify
  if(sum != 3825.0) return 4;


  sum = 50;
#pragma acc parallel loop reduction(+:sum) vector_length(32)
  for(i=0;i<100;i++) sum += array[i];
  //vefify
  if(sum != 5000.0) return 5;

  sum = 50;
#pragma acc parallel loop reduction(+:sum) vector_length(512)
  for(i=0;i<100;i++) sum += array[i];
  //vefify
  if(sum != 5000.0) return 6;
  
  return 0;
}
