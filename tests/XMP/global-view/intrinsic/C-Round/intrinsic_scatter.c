#include <stdio.h>
#include <stdlib.h>
#include "xmp.h"
extern void xmp_scatter(void *x_d, void *a_d, ... );

#pragma xmp nodes p(*)
#pragma xmp template t(0:15)
#pragma xmp distribute t(block) onto p 
  int x[16],a[16],idx[16];
#pragma xmp align a[i] with t(i) 
#pragma xmp align idx[i] with t(i) 
#pragma xmp align x[i] with t(i) 

int main()
{
  int i,result = 0;
  int adash[16],xdash[16],idxdash[16];
  for(i=0;i<16;i++)
    adash[i]=i;
  idxdash[0]=0;
  idxdash[1]=2;
  idxdash[2]=1;
  idxdash[3]=5;
  idxdash[4]=4;
  idxdash[5]=3;
  idxdash[6]=9;
  idxdash[7]=8;
  idxdash[8]=7;
  idxdash[9]=6;
  idxdash[10]=15;
  idxdash[11]=14;
  idxdash[12]=13;
  idxdash[13]=12;
  idxdash[14]=11;
  idxdash[15]=10;
  for(i=0;i<16;i++)
    xdash[idxdash[i]]=adash[i];


#pragma xmp loop on t(i)
  for(i=0;i<16;i++)
    a[i]=i;

#pragma xmp loop on t(i)
  for(i=0;i<16;i++)
    idx[i]=i;

#pragma xmp task on p(1)
    {
      idx[0]=0;
      idx[1]=2;
      idx[2]=1;
      idx[3]=5;
    }
#pragma xmp task on p(2)
    {
      idx[4]=4;
      idx[5]=3;
      idx[6]=9;
      idx[7]=8;
    }
#pragma xmp task on p(3)
    {
      idx[8]=7;
      idx[9]=6;
      idx[10]=15;
      idx[11]=14;
    }
#pragma xmp task on p(4)
    {
      idx[12]=13;
      idx[13]=12;
      idx[14]=11;
      idx[15]=10;
    }
#pragma xmp loop on t(i)
  for(i=0;i<16;i++)
    x[i]=0;

  xmp_scatter(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(idx));

#pragma xmp loop (i) on t(i)
  for(i=1;i<16;i++)
    if(x[i]!=xdash[i])
      result =-1;



#pragma xmp reduction(+:result)

#pragma xmp task on p(1)
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
