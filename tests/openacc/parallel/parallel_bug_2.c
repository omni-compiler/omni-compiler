int main()
{
  int a[10][10][10];

  int i,j,k;
#pragma acc parallel loop gang private(j)
  for(i=0;i<10;i++){
    for(j=0;j<10;j++){
#pragma acc loop vector
      for(k=0;k<10;k++){
	int tmp = j;
	a[i][tmp][k] = i*100+tmp*10+k;
	if(k == 0){
	  int tmp2 = j;
	  a[i][tmp2][k]++;
	}
      }
    }
  }

  for(i=0;i<10;i++){
    for(j=0;j<10;j++){
      for(k=0;k<10;k++){
	if(k == 0){
	  if(a[i][j][k] != i*100+j*10+k+1){
	    return 1;
	  }
	}else{
	  if(a[i][j][k] != i*100+j*10+k){
	    return 1;
	  }
	}
      }
    }
  }

  return 0;
}
