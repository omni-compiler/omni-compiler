#include "xmp.h"
#include <stdio.h>
#include <stdlib.h>

extern void xmp_pack(void *v_p, void *a_p, void *m_p);
extern void xmp_unpack(void *a_p, void *v_p, void *m_p);

#pragma xmp nodes p(2,2)
#pragma xmp nodes q(4)
#pragma xmp template tp(0:9,0:9)
#pragma xmp template tq(0:99)
#pragma xmp distribute tp(block,block) onto p
#pragma xmp distribute tq(block) onto q
  int a[10][10],adash[10][10],mask[10][10],v[100],vdash[100];
#pragma xmp align a[i][j] with tp(i,j)
#pragma xmp align mask[i][j] with tp(i,j)
#pragma xmp align v[i] with tq(i)

int main()
{

  int i,j;
  for(i=0;i<100;i++)
    if(i<50)
      vdash[i]=i*2;
    else
      vdash[i]=0;

  for(i=0;i<10;i++)
    for(j=0;j<10;j++)
      if(j%2==1)
	adash[i][j]=i*10+j-1;
      else
	adash[i][j]=0;

#pragma xmp loop (i,j) on tp(i,j)
  for(i=0;i<10;i++)
    for(j=0;j<10;j++)
      {
	if(j%2==0)
	  mask[i][j]=1;
	else
	  mask[i][j]=0;
      }

#pragma xmp loop (i,j) on tp(i,j)
  for(i=0;i<10;i++)
    for(j=0;j<10;j++)
      {
	a[i][j]=i*10+j;
      }

#pragma xmp loop (i) on tq(i)
  for(i=0;i<100;i++)
    {
      v[i]=0;
    }
  xmp_pack(xmp_desc_of(v),xmp_desc_of(a),xmp_desc_of(mask));


#pragma xmp loop (i,j) on tp(i,j)
  for(i=0;i<10;i++)
    for(j=0;j<10;j++)
      {
	a[i][j]=0;
	if(j%2==0)
	  mask[i][j]=0;
	else
	  mask[i][j]=1;
      }

  xmp_unpack(xmp_desc_of(a),xmp_desc_of(v),xmp_desc_of(mask));

  int result = 0;
#pragma xmp loop (i) on tq(i)
  for(i=0;i<100;i++)
    if(v[i]!=vdash[i])
      result =-1;

#pragma xmp reduction(+:result)

#pragma xmp task on p(1,1)
  {
    if (result == 0){
      printf("PASS\n");
    }
    else{
      printf("ERROR\n");
      exit(1);
    }
  }

#pragma xmp loop (i,j) on tp(i,j)
  for(i=0;i<10;i++)
    for(j=0;j<10;j++)
      if(a[i][j]!=adash[i][j])
        result =-1;



#pragma xmp reduction(+:result)

#pragma xmp task on p(1,1)
  {
    if (result == 0){
      printf("PASS\n");
    }
    else{
      printf("ERROR\n");
      exit(1);
    }
  }

  return 0;
}
