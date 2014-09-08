int main()
{
  int int_array[100];
  long long longlong_array[200];
  int i;
  long long j;

#pragma acc data copy(int_array[20:40]) copyout(longlong_array[100:])
  {
#pragma acc parallel loop
	for(i=20;i<60;i++){
	  int_array[i] = i;
	}
#pragma acc parallel loop
	for(j=100;j<200;j++){
	  longlong_array[j] = j;
	}
  }

  //verify
  for(i=20;i<60;i++){
	if(int_array[i] != i) return 1;
  }

  for(j=100;j<200;j++){
	if(longlong_array[j] != j) return 2;
  }

  return 0;
}
