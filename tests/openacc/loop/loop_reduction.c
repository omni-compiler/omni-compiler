int main()
{
  int i,j,k;
  double sum0, sum1;

  //2 var reduction
  sum0 = 123;
  sum1 = 456;
#pragma acc parallel
#pragma acc loop reduction(+:sum0, sum1)
  for(i=0;i<1000000;i++){
    sum0 += 1.0;
    sum1 += i;
  }
  //verify
  if(sum0 != 1000123.0 || sum1 != 499999500456.0){
    return 1;
  }


  sum0 = 1212;
#pragma acc parallel private(i)
  {
    for(i=0;i<10;i++){
#pragma acc loop reduction(+:sum0) gang vector
      for(j=0;j<100000;j++){
	sum0 += j;
      }
    }
  }

  if(sum0 != 49999501212.0){
    return 2;
  }


  sum0 = 1212;
#pragma acc parallel private(i) num_gangs(16)
  {
    for(i=0;i<10;i++){
#pragma acc loop reduction(+:sum0) gang vector
      for(j=0;j<100000;j++){
	sum0 += j;
      }
    }
  }

  if(sum0 != 49999501212.0){
    //printf("%lf\n",sum0);
    return 3;
  }

  sum0 = 1212;
#pragma acc parallel num_gangs(10)
  {
#pragma acc loop reduction(+:sum0) gang
    for(i=0;i<100;i++){
      double tmp = 0;
#pragma acc loop reduction(+:tmp) vector
      for(j=0;j<10000;j++){
	tmp += j;
      }
      sum0 += tmp;
    }
  }

  if(sum0 != 4999501212.0){
    //printf("%lf\n",sum0);
    return 4;
  }
  

  sum0 = 1212;
#pragma acc parallel num_gangs(10)
  {
#pragma acc loop reduction(+:sum0) gang
    for(i=0;i<100;i++){
#pragma acc loop reduction(+:sum0) vector
      for(j=0;j<10000;j++){
	sum0 += j;
      }
    }
  }

  if(sum0 != 4999501212.0){
    //printf("%lf\n",sum0);
    return 5;
  }

  return 0;
}
