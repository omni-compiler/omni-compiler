#define NAMELEN 25
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern int chk_int(char name[], int ierr);

void gmove_cp_4a4t_b_bc(){

char name[NAMELEN]="gmove_cp_4a4t_b_bc";
int n=8;
int a[n][n][n][n],b[n][n][n][n];
#pragma xmp nodes p[2][2][2][2]
#pragma xmp template tx[n][n][n][n]
#pragma xmp template ty[n][n][n][n]
#pragma xmp distribute tx[block][block][block][block] onto p
#pragma xmp distribute ty[cyclic(2)][cyclic(2)][cyclic(2)][cyclic(2)] onto p
#pragma xmp align a[i0][i1][i2][i3] with tx[i3][i2][i1][i0]
#pragma xmp align b[i0][i1][i2][i3] with ty[i3][i2][i1][i0]

  int i0,i1,i2,i3,ierr=0;

#pragma xmp loop on tx[i3][i2][i1][i0]
  for (i3=0;i3<n;i3++){
    for (i2=0;i2<n;i2++){
      for (i1=0;i1<n;i1++){
        for (i0=0;i0<n;i0++){
          a[i0][i1][i2][i3]=(i0+1)+(i1+1)+(i2+1)+(i3+1);
        }
      }
    }
  }

#pragma xmp loop on ty[i3][i2][i1][i0]
  for (i3=0;i3<n;i3++){
    for (i2=0;i2<n;i2++){
      for (i1=0;i1<n;i1++){
        for (i0=0;i0<n;i0++){
          b[i0][i1][i2][i3]=0;
        }
      }
    }
  }

#pragma xmp gmove
  b[1:4][1:4][1:4][1:4]=a[4:4][4:4][4:4][4:4];

#pragma xmp loop on ty[i3][i2][i1][i0]
  for (i3=1;i3<5;i3++){
    for (i2=1;i2<5;i2++){
      for (i1=1;i1<5;i1++){
        for (i0=1;i0<5;i0++){
          ierr=ierr+abs(b[i0][i1][i2][i3]-(i0+4)-(i1+4)-(i2+4)-(i3+4));
        }
      }
    }
  }

#pragma xmp reduction (+:ierr)
  chk_int(name, ierr);

}

int main(){

  gmove_cp_4a4t_b_bc();

  return 0;

}
