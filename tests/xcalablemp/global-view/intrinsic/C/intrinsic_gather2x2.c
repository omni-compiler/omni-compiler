#include <stdio.h>
#include <stdlib.h>
#include "xmp.h"
extern void xmp_gather(void *x_d, void *a_d, ... );

int main()
{
#pragma xmp nodes p(2,2)
#pragma xmp template t(0:3,0:3)
#pragma xmp distribute t(block,block) onto p 
int x[4][4],a[4][4],idx0[4][4],idx1[4][4];
#pragma xmp align a[i][j] with t(i,j) 
#pragma xmp align idx0[i][j] with t(i,j) 
#pragma xmp align idx1[i][j] with t(i,j) 
#pragma xmp align x[i][j] with t(i,j) 
int i,j;

#pragma xmp loop (i,j) on t(i,j)
for(i=0;i<4;i++)
 for(j=0;j<4;j++)
 a[i][j]=i*4+j;
xmp_barrier();

/*
#pragma xmp loop (i,j) on t(i,j)
for(i=0;i<4;i++)
 for(j=0;j<4;j++)
 printf("nodes=%d a[%d][%d]=%d\n",xmp_node_num(),i,j,a[i][j]);
xmp_barrier();
*/

#pragma xmp loop (i,j) on t(i,j)
for(i=0;i<4;i++)
 for(j=0;j<4;j++)
{
idx0[i][j]=i;
idx1[i][j]=j;
}
xmp_barrier();

#pragma xmp loop (i,j) on t(i,j)
for(i=0;i<4;i++)
 for(j=0;j<4;j++)
x[i][j]=0;
xmp_barrier();


/*
#pragma xmp loop (i,j) on t(i,j)
for(i=0;i<4;i++)
 for(j=0;j<4;j++)
 printf("nodes=%d idx[%d][%d]=%d idx[%d][%d]=%d\n",xmp_node_num(),i,j,idx0[i][j],i,j,idx1[i][j]);
xmp_barrier();
*/

xmp_gather(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(idx0),xmp_desc_of(idx1));

xmp_barrier();

/*
#pragma xmp loop (i,j) on t(i,j)
for(i=0;i<4;i++)
 for(j=0;j<4;j++)
 printf("nodes=%d x[%d][%d]=%d\n",xmp_node_num(),i,j,x[i][j]);
xmp_barrier();
*/

int result = 0;
#pragma xmp loop (i,j) on t(i,j)
for(i=0;i<4;i++)
 for(j=0;j<4;j++)
  if(x[i][j]!=a[i][j])
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
