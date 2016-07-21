#define L 3
#define M 10
#define N 25

int main()
{
  int array[L*M*N];
  int i,j,k;

  //gang collapse
#pragma acc parallel
#pragma acc loop collapse(2) gang
  for(i=0;i<L;i++){
    for(j=0;j<M;j++){
#pragma acc loop vector
      for(k=0;k<N;k++){
	array[(i*M+j)*N + k] = i*3 + j*5 + k*11;
      }
    }
  }

  //verify
  for(i=0;i<L;i++){
    for(j=0;j<M;j++){
      for(k=0;k<N;k++){
	if(array[(i*M+j)*N + k] != i*3 + j*5 + k*11){
	  return 1;
	}
      }
    }
  }


  //vector collapse
#pragma acc parallel
#pragma acc loop gang
  for(i=0;i<L;i++){
#pragma acc loop collapse(2) vector
    for(j=0;j<M;j++){
      for(k=0;k<N;k++){
	array[(i*M+j)*N + k] = i*7 + j*11 + k*23;
      }
    }
  }

  //verify
  for(i=0;i<L;i++){
    for(j=0;j<M;j++){
      for(k=0;k<N;k++){
	if(array[(i*M+j)*N + k] != i*7 + j*11 + k*23){
	  return 2;
	}
      }
    }
  }


  //gang and vector collapse
#pragma acc parallel
#pragma acc loop collapse(3) gang vector
  for(i=0;i<L;i++){
    for(j=0;j<M;j++){
      for(k=0;k<N;k++){
	array[(i*M+j)*N + k] = i*17 + j*3 + k*2;
      }
    }
  }

  //verify
  for(i=0;i<L;i++){
    for(j=0;j<M;j++){
      for(k=0;k<N;k++){
	if(array[(i*M+j)*N + k] != i*17 + j*3 + k*2){
	  return 3;
	}
      }
    }
  }


  int array3d[L][M][N];
  //gang and vector collapse with loop initial declarations
#pragma acc parallel
#pragma acc loop collapse(3)
  for(int s = 0; s < L; s++){
    for(int t = 0; t < M; t++){
      for(int u = 0; u < N; u++){
	array3d[s][t][u] = s*3 + t*5 + u*11;
      }
    }
  }

  //verify
  for(int s = 0; s < L; s++){
    for(int t = 0; t < M; t++){
      for(int u = 0; u < N; u++){
	if(array3d[s][t][u] != s*3 + t*5 + u*11){
	  return 4;
	}
      }
    }
  }

  return 0;
}
