#define NAMELEN 25
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern int chk_int(char name[], int ierr);

void gmove_bca_2a3t_b2_subcomm(){

char name[NAMELEN]="gmove_bca_2a3t_b2_subcomm";
int n=4;
int a[n][n],b[n][n];
#pragma xmp nodes p(8)
#pragma xmp nodes p1(2,2)=p(1:4)
#pragma xmp nodes p2(2,2)=p(5:8)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp template ty(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(block,*,block) onto p1
#pragma xmp distribute ty(*,block,block) onto p2
#pragma xmp align a[i0][*] with tx(i0,*,*)
#pragma xmp align b[*][i2] with ty(*,*,i2)

  int i0,i1,i2,ierr;

#pragma xmp loop (i0) on tx(i0,*,*)
  for(i0=0;i0<n;i0++){
    for(i2=0;i2<n;i2++){
      a[i0][i2]=i0+i2+1;
    }
  }

  for(i1=0;i1<n;i1++){
#pragma xmp loop (i2) on ty(*,*,i2)
    for(i2=0;i2<n;i2++){
      b[i1][i2]=0;
    }
  }

#pragma xmp gmove
  b[:][:]=a[:][:];

  ierr=0;
  for(i1=0;i1<n;i1++){
#pragma xmp loop (i2) on ty(*,*,i2)
    for(i2=0;i2<n;i2++){
      ierr=ierr+abs(b[i1][i2]-i1-i2-1);
    }
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

void gmove_bca_2a3t_b(){

char name[NAMELEN]="gmove_bca_2a3t_b";
int n=8;
int a[n][n],b[n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(block,block,block) onto p
#pragma xmp align a[i0][i1] with tx(i0,i1,*)
#pragma xmp align b[i1][i2] with tx(*,i1,i2)

  int i0,i1,i2,ierr;

#pragma xmp loop (i0,i1) on tx(i0,i1,*)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      a[i0][i1]=i0+i1+1;
    }
  }

#pragma xmp loop (i1,i2) on tx(*,i1,i2)
  for(i1=0;i1<n;i1++){
    for(i2=0;i2<n;i2++){
      b[i1][i2]=0;
    }
  }

#pragma xmp gmove
  b[:][:]=a[:][:];

  ierr=0;
#pragma xmp loop (i1,i2) on tx(*,i1,i2)
  for(i1=0;i1<n;i1++){
    for(i2=0;i2<n;i2++){
      ierr=ierr+abs(b[i1][i2]-i1-i2-1);
    }
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

void gmove_bca_3a3t_b2(){

char name[NAMELEN]="gmove_bca_3a3t_b2";
int n=4;
int a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(block,block,block) onto p
#pragma xmp align a[i0][i1][i2] with tx(i0,i1,i2)
#pragma xmp align b[*][i1][i2] with tx(*,i1,i2)

  int i0,i1,i2,ierr;

#pragma xmp loop (i0,i1,i2) on tx(i0,i1,i2)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        a[i0][i1][i2]=i0+i1+i2+1;
      }
    }
  }

#pragma xmp loop (i1,i2) on tx(*,i1,i2)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        b[i0][i1][i2]=0;
      }
    }
  }

#pragma xmp gmove
  b[:][:][:]=a[:][:][:];

  ierr=0;
#pragma xmp loop (i1,i2) on tx(*,i1,i2)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        ierr=ierr+abs(b[i0][i1][i2]-i0-i1-i2-1);
      }
    }
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

void gmove_bca_3a3t_b(){

char name[NAMELEN]="gmove_bca_3a3t_b";
int n=4;
int a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(block,block,block) onto p
#pragma xmp align a[i0][i1][i2] with tx(i0,i1,i2)
#pragma xmp align b[*][i1][i2] with tx(*,i1,i2)

  int i0,i1,i2,ierr;

#pragma xmp loop (i0,i1,i2) on tx(i0,i1,i2)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        a[i0][i1][i2]=i0+i1+i2+1;
      }
    }
  }

#pragma xmp loop (i1,i2) on tx(*,i1,i2)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        b[i0][i1][i2]=0;
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n]=a[0:n][0:n][0:n];

  ierr=0;
#pragma xmp loop (i1,i2) on tx(*,i1,i2)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        ierr=ierr+abs(b[i0][i1][i2]-i0-i1-i2-1);
      }
    }
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

void gmove_bca_3a3t_b_s(){

char name[NAMELEN]="gmove_bca_3a3t_b_s";
int n=8;
int a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp nodes q(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp template ty(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(block,block,block) onto p
#pragma xmp distribute ty(block,block,block) onto q
#pragma xmp align a[i0][i1][i2] with tx(i0,i1,i2)
#pragma xmp align b[*][i1][i2] with ty(*,i1,i2)
#pragma xmp shadow a[0][0][1]
#pragma xmp shadow b[0][0][1]

  int i0,i1,i2,ierr;

#pragma xmp loop (i0,i1,i2) on tx(i0,i1,i2)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        a[i0][i1][i2]=i0+i1+i2+1;
      }
    }
  }

#pragma xmp loop (i1,i2) on ty(*,i1,i2)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        b[i0][i1][i2]=0;
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n]=a[0:n][0:n][0:n];

  ierr=0;
#pragma xmp loop (i1,i2) on ty(*,i1,i2)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        ierr=ierr+abs(b[i0][i1][i2]-i0-i1-i2-1);
      }
    }
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

void gmove_bca_3a4t_b(){

char name[NAMELEN]="gmove_bca_3a4t_b";
int n=8;
int a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1,0:n-1)
#pragma xmp template ty(0:n-1,0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(block,block,*,block) onto p
#pragma xmp distribute ty(block,*,block,block) onto p
#pragma xmp align a[*][i1][i3] with tx(*,i1,*,i3)
#pragma xmp align b[i0][i2][*] with ty(i0,*,i2,*)

  int i0,i1,i2,i3,ierr;

  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1,i3) on tx(*,i1,*,i3)
    for(i1=0;i1<n;i1++){
      for(i3=0;i3<n;i3++){
        a[i0][i1][i3]=i0+i1+i3+1;
      }
    }
  }

#pragma xmp loop (i0,i2) on ty(i0,*,i2,*)
  for(i0=0;i0<n;i0++){
    for(i2=0;i2<n;i2++){
      for(i3=0;i3<n;i3++){
        b[i0][i2][i3]=0;
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n]=a[0:n][0:n][0:n];

  ierr=0;
#pragma xmp loop (i0,i2) on ty(i0,*,i2,*)
  for(i0=0;i0<n;i0++){
    for(i2=0;i2<n;i2++){
      for(i3=0;i3<n;i3++){
        ierr=ierr+abs(b[i0][i2][i3]-i0-i2-i3-1);
      }
    }
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

int main(){

  gmove_bca_2a3t_b2_subcomm();
  gmove_bca_2a3t_b();
  gmove_bca_3a3t_b2();
  gmove_bca_3a3t_b();
  gmove_bca_3a3t_b_s();
  gmove_bca_3a4t_b();

  return 0;

}
