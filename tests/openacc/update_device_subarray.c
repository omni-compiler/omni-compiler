#include<stdio.h>
#include<stdlib.h>

typedef struct quad{
  int a;
  int b;
  int c;
  int d;
}quad;

#ifdef CHAR
#define TYPE char
#define ZEROVAL (0)
#define SETVAL (i*11 + j*13 + k*17 + l*19)%128
#endif

#ifdef SHORT
#define TYPE short
#define ZEROVAL (0)
#define SETVAL (i*11 + j*13 + k*17 + l*19)%32767
#endif

#ifdef INT
#define TYPE int
#define ZEROVAL (0)
#define SETVAL (i*11 + j*13 + k*17 + l*19)
#endif

#ifdef FLOAT
#define TYPE float
#define ZEROVAL (0.0)
#define SETVAL (i*11 + j*13 + k*17 + l*19)
#endif

#ifdef DOUBLE
#define TYPE double
#define ZEROVAL (0.0)
#define SETVAL (double)(i*11 + j*13 + k*17 + l*19)
#endif

#ifndef TYPE
#define TYPE quad
#define ZEROVAL (quad){0,0,0,0}
#define SETVAL (quad){i,j,k,l}
#define STRUCT
#endif

/*
#define I 3
#define J 5
#define K 7
#define L 9
*/
#define I 11
#define J 19
#define K 23
#define L 29

int test_update_device(TYPE *a, int i_low, int i_len, int j_low, int j_len, int k_low, int k_len, int l_low, int l_len);
void get_range(int range[2], int size, int type);

int main()
{
  int i,j,k,l;
  TYPE *a;

  a = (TYPE*)malloc(sizeof(TYPE)*I*J*K*L);

  
#pragma acc data create(a[0:I*J*K*L])
  {
    for(i = 0; i < I; i++){
      for(j = 0; j < J; j++){
	for(k = 0; k < K; k++){
	  for(l = 0; l < L; l++){
	    a[(((i*J)+j)*K+k)*L+l] = (SETVAL);
	  }
	}
      }
    }

    //tests
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	for(k=0;k<3;k++){
	  for(l=0;l<3;l++){
	    int range_i[2],range_j[2],range_k[2],range_l[2];
	    get_range(range_i, I, i);
	    get_range(range_j, J, j);
	    get_range(range_k, K, k);
	    get_range(range_l, L, l);
	    if(test_update_device(a, range_i[0], range_i[1], range_j[0], range_j[1], range_k[0], range_k[1], range_l[0], range_l[1]) ){
	      printf("update device([%d:%d][%d:%d][%d:%d][%d:%d]) failed\n", range_i[0], range_i[1], range_j[0], range_j[1], range_k[0], range_k[1], range_l[0], range_l[1]);
	      exit(1);
	    }
	  }
	}
      }
    }
  }
  
  return 0;
}

int test_update_device(TYPE *a_in, int i_low, int i_len, int j_low, int j_len, int k_low, int k_len, int l_low, int l_len)
{
  int i,j,k,l;
  int flag = 0;
  TYPE (*a)[J][K][L] = (TYPE (*)[J][K][L])a_in;
  
#pragma acc data present(a[0:I][0:J][0:K][0:L])
  {
    //set zero
#pragma acc parallel loop collapse(4)
    for(i = 0; i < I; i++)
      for(j = 0; j < J; j++)
	for(k = 0; k < K; k++)
	  for(l = 0; l < L; l++)
	    a[i][j][k][l] = (ZEROVAL);
    

#pragma acc update device(a[i_low:i_len][j_low:j_len][k_low:k_len][l_low:l_len])

    //check
#pragma acc parallel loop collapse(4) reduction(+:flag)
    for(i = 0; i < I; i++){
      for(j = 0; j < J; j++){
	for(k = 0; k < K; k++){
	  for(l = 0; l < L; l++){
	    TYPE v = a[i][j][k][l];
	    if( (i>=i_low&&i<i_low+i_len) && (j>=j_low&&j<j_low+j_len) && (k>=k_low&&k<k_low+k_len) && (l>=l_low&&l<l_low+l_len) ){
#ifndef STRUCT
	      if(v != (SETVAL)){
#else
	      if(v.a != i || v.b != j || v.c != k || v.d != l){
#endif
		flag+= 1;
	      }
	    }else{
#ifndef STRUCT
	      if(v != (ZEROVAL)){
#else
	      if(v.a != 0 || v.b != 0 || v.c != 0 || v.d != 0){
#endif
	        flag+=1;
	      }
	    }
	  }
	}
      }
    }
  }

  return flag;
}

void get_range(int range[2], int size, int type)
{
  switch(type){
  case 0: //full
    range[0] = 0;
    range[1] = size;
    break;
  case 1: //len=1
    range[0] = size/2;
    range[1] = 1;
    break;
  case 2: //part
    range[0] = 1;
    range[1] = size - 2;
    break;
  default:
    printf("invalid argument\n");
  }
}
