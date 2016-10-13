  program decl
!!     include "xmp_coarray.h"
    real, allocatable :: a(:)[:]
    integer nerr = 0
    integer me,my

    me = this_image()

    if (allocated(a)) then
       nerr = nerr+1
       write(*,200) me, "allocated(a) sould be false"
    endif

    allocate(a(321)[3:*])
    if (size(a).ne.321) then
       nerr = nerr+1
       write(*,201) me, 321, size(a)
    endif

    syncall
    !! a(i)[(/4,5,3/)]=i*1.0+(/1,2,3/)*0.1
    do i=1,300
       a(i)[mod(me,3)+3]=i*1.0+me*0.1
    enddo
    syncall

    do i=1,300
       do k=1,3
          my=mod(k,3)+3
          if (me+2==my) then
             val=i*1.0+k*0.1
          endif
       enddo
       if (a(i).ne.val) then
          nerr=nerr+1
          write(*,202) me, i, val, a(i)
       endif
    enddo

200 format("[",i0,"] ",a) 
201 format("[",i0,"] size(a) must be ",i0," but ",i0)
202 format("[",i0,"] a(",i0,") must be ",f10.6," but ",f10.6)

!!! restriction due to OMNI   
!!    if (allocated(s)) then
!!       nerr = nerr+1
!!       write(*,100) me, "allocated(s) sould be false"
!!    endif

    if (nerr==0) then
       write(*,100) me, "OK"
    else
       write(*,101) me, "NG", nerr
    endif

100 format("[",i0,"] ",a) 
101 format("[",i0,"] ",a," nerr=",i0) 

  end program decl
