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

void sub(TYPE *a1, TYPE (*a2)[20], TYPE (*a3)[20][30]){
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
  TYPE *c1, (*c2)[20], (*c3)[20][30];

  c1 = (TYPE*)malloc(sizeof(TYPE)*100);
  c2 = (TYPE (*)[20])malloc(sizeof(TYPE)*10*20);
  c3 = (TYPE (*)[20][30])malloc(sizeof(TYPE)*10*20*30);

#pragma acc data copy(c1[0:100], c2[0:10][0:20], c3[0:10][0:20][0:30])
  {
    sub(c1, c2, c3);
  }

  return 0;
}
