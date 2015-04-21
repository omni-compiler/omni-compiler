  module mmm
    include "xmp_coarray.h"
    real*8 da(10,10)[3,*]

  contains
    subroutine abc
      continue
    end subroutine abc
    subroutine mmm_abc_def
      continue
    end subroutine mmm_abc_def
  end module mmm

  use mmm

  me = this_image()

  !----------------------------- init
  pi = 3.1415926535897946
  do j=1,10
     do i=1,10
        da(i,j) = me*sin(pi/i) + cos(pi/j)
     enddo
  enddo
  sync all
  !----------------------------- exec
  if (me==2) then
     da(3,:) = da(:,5)[3,1]
  else if (me==1) then
     da(:,7) = da(2,:)[3,1]
  endif
  syncall
  !----------------------------- check
  nerr=0
  eps=0.00001
  do j=1,10
     do i=1,10
        if (me==2.and.i==3) then
           val=3*sin(pi/j)+cos(pi/5)
        else if (me==1.and.j==7) then
           val=3*sin(pi/2)+cos(pi/i)
        else
           val=me*sin(pi/i)+cos(pi/j)
        endif

        if (val-eps < da(i,j) .and. da(i,j) < val+eps) then
           continue
        else
           nerr=nerr+1
           write(*,101) i,j,me,da(i,j),val
        endif

     enddo
  enddo

  if (nerr==0) then 
     print '("[",i0,"] OK")', me
  else
     print '("[",i0,"] number of NGs: ",i0)', me, nerr
  end if

101 format ("da(",i0,",",i0")[",i0,"]=",f8.6," should be ",f8.6)
 end

