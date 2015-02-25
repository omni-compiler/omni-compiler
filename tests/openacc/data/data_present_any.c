#include <stdlib.h>

typedef struct pair{
  int a;
  int b;
}pair;

#ifdef CHAR
#define TYPE char
#define SETVAL (123)
#endif

#ifdef INT
#define TYPE int
#define SETVAL (0)
#endif

#ifdef FLOAT
#define TYPE float
#define SETVAL (123.0)
#endif

#ifdef DOUBLE
#define TYPE double
#define SETVAL (12345.0)
#endif

#ifndef TYPE
#define TYPE pair
#define SETVAL (pair){1,2}
#endif

void sub(TYPE *a1, TYPE *a2_, TYPE *a3_){
  TYPE (*a2)[20] = (TYPE (*)[20])a2_;
  TYPE (*a3)[20][30] = (TYPE (*)[20][30])a3_;
#pragma acc data present(a1[0:100], a2[0:10][0:20], a3[:10][:20][:30])
  {
    int i,j,k;
#pragma acc parallel loop
    for(i=0;i<100;i++){
      a1[i] = SETVAL;
    }

#pragma acc parallel loop collapse(2)
    for(i=0;i<10;i++){
      for(j=0;j<20;j++){
	a2[i][j] = SETVAL;
      }
    }
    
#pragma acc parallel loop collapse(3)
    for(i=0;i<10;i++){
      for(j=0;j<20;j++){
	for(k=0;k<30;k++){
	  a3[i][j][k] = SETVAL;
	}
      }
    }
  }
}

int main()
{
  TYPE a1[100];
  TYPE a2[10][20];
  TYPE a3[10][20][30];
  
  TYPE *b1, *b2, *b3;

  b1 = (TYPE*)malloc(sizeof(TYPE)*100);
  b2 = (TYPE*)malloc(sizeof(TYPE)*10*20);
  b3 = (TYPE*)malloc(sizeof(TYPE)*10*20*30);

  if(b1 == NULL || b2 == NULL || b3 == NULL){
    return 1;
  }
  
#pragma acc data copy(a1, a2, a3, b1[0:100], b2[0:10*20], b3[0:10*20*30])
  {
    sub((TYPE*)a1, (TYPE*)a2, (TYPE*)a3);
    sub(b1, b2, b3);
  }

  return 0;
}
