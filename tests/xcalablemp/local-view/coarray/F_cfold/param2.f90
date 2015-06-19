  include "xmp_coarray.h"
  parameter(n1=10)
  integer, parameter:: n2=3

  integer*8 a(n2:n1+1,10)[*]

  !----------------------------------- init
  me = this_image()
  a=-999

  if (me==1) then
     do j=1,10
        do i=n2,n1+1
           a(i,j)=i*100+j
        enddo
     enddo
  end if

  sync all

  !----------------------------------- exec
  if (me==3) then
     a=a[1]
  end if

  sync all

  !----------------------------------- check
  nerr=0
  do j=1,10
     do i=n2,n1+1
        if (me==1.or.me==3) then
           if (a(i,j).ne.i*100+j) then
              nerr=nerr+1
           end if
        else
           if (a(i,j).ne.-999) then
              nerr=nerr+1
           end if
        end if
     end do
  end do

  if (nerr==0) then 
     print '("[",i0,"] OK")', me
  else
     print '("[",i0,"] number of NGs: ",i0)', me, nerr
  end if

101 format ("a(",i0,",",i0,")[",i0,"]=",i0," should be ",i0)

  end
