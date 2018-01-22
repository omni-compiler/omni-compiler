!!   include "xmp_coarray.h"
  integer a1d1(10)[*], tmp(10)
  intrinsic max

!--------- init
  me = xmp_node_num()
  do i=1,10
     a1d1(i) = i*me+max(-1,0)
  enddo
  tmp = 0
  sync all

!--------- exec
!!  tmp(2:6) = a1d1(2:a1d1(3)[2])[a1d1(1)[3]]
  nn=a1d1(3)[2]
  sync all
  write(*,*) "me,nn=",me,nn
  tmp(2:6) = a1d1(2:nn)[3]
  sync all

!--------- check
  nerr=0
  do i=1,10
     if (i.ge.2.and.i.le.6) then
        ival = i*3
     else
        ival = 0
     endif
     if (tmp(i).ne.ival) then
        write(*,101) i,me,tmp(i),ival
        nerr=nerr+1
     end if
  enddo

101 format ("tmp(",i0,")[",i0,"]=",i0," should be ",i0)

  if (nerr==0) then 
     print '("[",i0,"] OK")', me
  else
     print '("[",i0,"] number of NGs: ",i0)', me, nerr
  end if

end program

