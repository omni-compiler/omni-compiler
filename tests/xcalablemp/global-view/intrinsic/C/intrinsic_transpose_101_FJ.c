#include <stdio.h>
#include <xmp.h>
extern int chk_int(int ierr);
extern void xmp_transpose(void *dst_d, void *src_d, int opt);

int main(){

  //  int ret;
  int ierr,error,i,j/*,k*/;
  int a[4][4], b[4][4], rc[4][4];

#pragma xmp nodes p(16)
#pragma xmp nodes pa(4,4)=p(1:16)
#pragma xmp template ta(0:3,0:3)
#pragma xmp distribute ta(block,block) onto pa
#pragma xmp align a[i][j] with ta(i,j)
#pragma xmp align b[i][j] with ta(i,j)

  /* init */
#pragma xmp loop (i,j) on ta(i,j)
  for(i=0;i<4;i++){
    for(j=0;j<4;j++){
      b[i][j] = -2;
    }
  }

#pragma xmp loop (i,j) on ta(i,j)
  for(i=0;i<4;i++){
    for(j=0;j<4;j++){
      a[i][j] = -1;
    }
  }

  for(j=0;j<4;j++){
    for(i=0;i<4;i++){
      if(rc[j][i] != (i-1)*j){
  	ierr++;
      }
    }
  }


#pragma xmp loop (i,j) on ta(i,j)
  for(i=0;i<4;i++){
    for(j=0;j<4;j++){
      a[i][j] = i*(j-1);
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);


  error = 0;

#pragma xmp loop (i,j) on ta(i,j) reduction(+:ierr)
  for(i=0;i<4;i++){
    for(j=0;j<4;j++){
      if (b[i][j] != j*(i-1)){
	error++;
      }
    }
  }

   return chk_int(error);
}
